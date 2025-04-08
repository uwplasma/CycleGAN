import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import Generator  
from qsc import Qsc
from qsc.util import mu0, fourier_minimum, to_Fourier

def standardize(X, mean, std):
    return (X - mean) / std

def inverse_standardize(X_scaled, mean, std):
    return X_scaled * std + mean

def filter_generated_data(generated_csv, basic_out_csv, fourier_out_csv):
    generated_df = pd.read_csv(generated_csv)
    basic_valid = pd.DataFrame()         
    valid_with_fourier = pd.DataFrame()    

    
    for index, row in generated_df.iterrows():
        try:
            
            rc_values = [row['rc1'], row['rc2'], row['rc3']]
            zs_values = [row['zs1'], row['zs2'], row['zs3']]
            nfp_value = row['nfp']
            etabar_value = row['etabar']
            B2c_value = row['B2c']
            p2_value = row['p2']

            
            stel = Qsc(
                rc=[1.0] + rc_values,
                zs=[0.0] + zs_values,
                nfp=nfp_value,
                etabar=etabar_value,
                B2c=B2c_value,
                p2=p2_value,
                order='r2',
                nphi=51
            )

            
            axis_length    = stel.axis_length
            iota           = stel.iota
            max_elongation = stel.max_elongation
            min_L_grad_B   = stel.min_L_grad_B
            min_R0         = stel.min_R0
            r_singularity  = stel.r_singularity
            L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
            B20_variation  = stel.B20_variation
            beta           = -mu0 * p2_value * (stel.r_singularity ** 2) / (stel.B0 ** 2)
            DMerc_times_r2 = stel.DMerc_times_r2

            
            basic_condition = (
                axis_length > 0 and
                abs(iota) >= 0.2 and
                max_elongation <= 10 and
                abs(min_L_grad_B) >= 0.1 and
                abs(min_R0) >= 0.3 and
                r_singularity >= 0.05 and
                np.all(abs(L_grad_grad_B) >= 0.1) and
                B20_variation <= 5 and
                beta >= 0.0001 and
                DMerc_times_r2 > 0
            )

            if basic_condition:
                basic_valid = pd.concat([basic_valid, pd.DataFrame([row])], ignore_index=True)
                
                
                try:
                    R_2D, Z_2D, _ = stel.Frenet_to_cylindrical(r=0.1, ntheta=20)
                    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, stel.nfp, mpol=13, ntor=25, lasym=stel.lasym)
                    valid_with_fourier = pd.concat([valid_with_fourier, pd.DataFrame([row])], ignore_index=True)
                except Exception as e:
                    
                    pass

        except Exception as e:
            print(f"Row {index} raised an error: {e}")

    
    basic_valid.to_csv(basic_out_csv, index=False)
    valid_with_fourier.to_csv(fourier_out_csv, index=False)
    print(f"Saved basic valid data to {basic_out_csv}")
    print(f"Saved Fourier valid data to {fourier_out_csv}")

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_data = pd.read_csv(args.input_csv)
    
    X_new = new_data.iloc[:, :10].values

    
    scaler_X_mean = np.load('scaler_X_mean.npy')
    scaler_X_std  = np.load('scaler_X_std.npy')
    scaler_Y_mean = np.load('scaler_Y_mean.npy')
    scaler_Y_std  = np.load('scaler_Y_std.npy')

    
    X_new_scaled = standardize(X_new, scaler_X_mean, scaler_X_std)

    
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    
    input_dim = 10
    output_dim = 10
    G_A2B = Generator(input_dim, output_dim).to(device)
    G_A2B.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    G_A2B.eval()

    
    with torch.no_grad():
        generated_output = G_A2B(X_new_tensor).cpu().numpy()

    
    generated_output_inversed = inverse_standardize(generated_output, scaler_Y_mean, scaler_Y_std)

    
    generated_df = pd.DataFrame(generated_output_inversed, columns=new_data.columns[:10])

    if 'nfp' in generated_df.columns:
        generated_df['nfp'] = generated_df['nfp'].astype(int)

    
    generated_df.to_csv(args.generated_csv, index=False)

    print(f"Generated data saved to {args.generated_csv}")

    if args.filter:
        filter_generated_data(args.generated_csv, args.basic_out_csv, args.fourier_out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data using a trained model and optionally filter it")
    parser.add_argument("--input_csv", type=str, default="new_data.csv", help="Path to input CSV file with data to be transformed")
    parser.add_argument("--generated_csv", type=str, default="generated_output.csv", help="Path to save the generated data")
    parser.add_argument("--model_path", type=str, default="G_A2B_best_model.pth", help="Path to the trained generator model")
    parser.add_argument("--filter", type=bool, default=False, help="Whether to apply additional filtering to the generated data")
    parser.add_argument("--basic_out_csv", type=str, default="generated_output_basic.csv", help="Output CSV for basic valid data")
    parser.add_argument("--fourier_out_csv", type=str, default="generated_output_fourier.csv", help="Output CSV for Fourier valid data")
    
    args = parser.parse_args()
    main(args)
