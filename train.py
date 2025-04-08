import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from model import Generator, Discriminator


original_data = pd.read_csv('bad_stels.csv')
target_data = pd.read_csv('XGStels.csv')


X = original_data.iloc[:, :10].values
Y = target_data.iloc[:, :10].values


scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)


X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.15, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)


batch_size = 1024
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 10
output_dim = 10
G_A2B = Generator(input_dim, output_dim).to(device)
G_B2A = Generator(output_dim, input_dim).to(device)
D_A = Discriminator(input_dim).to(device)
D_B = Discriminator(output_dim).to(device)


adversarial_loss = nn.MSELoss()  
cycle_loss = nn.L1Loss()         


optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0001, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0001, betas=(0.5, 0.999))


fake_A_buffer = deque(maxlen=50)
fake_B_buffer = deque(maxlen=50)


epoch = 0
eval_interval = 10000 
patience = 10          
no_improve_count = 0
best_basic_ratio = 0.0
best_fourier_ratio = 0.0
best_generator_state = None


eval_steps = []
basic_ratio_list = []
fourier_ratio_list = []


while True:
    epoch += 1
    G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()
    num_batches = 0

    for real_A, real_B in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        real_A, real_B = real_A.to(device), real_B.to(device)
        current_batch_size = real_A.size(0)
        valid = torch.ones((current_batch_size, 1), device=device)
        fake_label = torch.zeros((current_batch_size, 1), device=device)
        
        
        optimizer_G.zero_grad()
        loss_id_A = cycle_loss(G_B2A(real_A), real_A)
        loss_id_B = cycle_loss(G_A2B(real_B), real_B)
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = adversarial_loss(D_B(fake_B), valid)
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = adversarial_loss(D_A(fake_A), valid)
        recov_A = G_B2A(fake_B)
        loss_cycle_A = cycle_loss(recov_A, real_A)
        recov_B = G_A2B(fake_A)
        loss_cycle_B = cycle_loss(recov_B, real_B)
        loss_G = (4 * loss_GAN_A2B + loss_GAN_B2A) + (loss_cycle_A + loss_cycle_B) + (loss_id_A + loss_id_B)
        loss_G.backward()
        optimizer_G.step()
        
        fake_A_buffer.append(fake_A.detach().cpu())
        fake_B_buffer.append(fake_B.detach().cpu())
        
        
        if num_batches % 2 == 0:
            
            optimizer_D_A.zero_grad()
            loss_real_A = adversarial_loss(D_A(real_A), valid)
            loss_fake_A = adversarial_loss(D_A(fake_A.detach()), fake_label)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            
            optimizer_D_B.zero_grad()
            loss_real_B = adversarial_loss(D_B(real_B), valid)
            loss_fake_B = adversarial_loss(D_B(fake_B.detach()), fake_label)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
        
        num_batches += 1

   
    if epoch % eval_interval == 0:
        G_A2B.eval()
        
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            generated_output = G_A2B(X_eval_tensor).cpu().numpy()
        
        
        generated_output = scaler_Y.inverse_transform(generated_output)
        generated_df = pd.DataFrame(generated_output, columns=target_data.columns[:10])
        
        
        from qsc import Qsc
        from qsc.util import mu0, fourier_minimum, to_Fourier
        
        basic_valid = pd.DataFrame()         
        valid_with_fourier = pd.DataFrame()    
        
        for index, row in generated_df.iterrows():
            try:
                
                label = [row['rc1'], row['rc2'], row['rc3'],
                         row['zs1'], row['zs2'], row['zs3'],
                         row['nfp'], row['etabar'], row['B2c'], row['p2']]
                rc_values = label[:3]
                zs_values = label[3:6]
                nfp_value = int(row['nfp'])
                etabar_value = label[7]
                B2c_value = row['B2c']
                p2_value = row['p2']
    
                
                stel = Qsc(
                    rc=[1.] + rc_values,
                    zs=[0.] + zs_values,
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
    
                
                if (axis_length > 0 and
                    abs(iota) >= 0.2 and
                    max_elongation <= 10 and
                    abs(min_L_grad_B) >= 0.1 and
                    abs(min_R0) >= 0.3 and
                    r_singularity >= 0.05 and
                    abs(L_grad_grad_B) >= 0.1 and
                    B20_variation <= 5 and
                    beta >= 0.0001 and
                    DMerc_times_r2 > 0):
                    
                    basic_valid = pd.concat([basic_valid, pd.DataFrame([row])], ignore_index=True)
                    
                    try:
                        R_2D, Z_2D, _ = stel.Frenet_to_cylindrical(r=0.1, ntheta=20)
                        RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, stel.nfp, mpol=13, ntor=25, lasym=stel.lasym)
                        valid_with_fourier = pd.concat([valid_with_fourier, pd.DataFrame([row])], ignore_index=True)
                    except Exception as e:
                        pass
    
            except Exception as e:
                print(f"Row {index} raised an error: {e}")
        
        
        accuracy_basic = len(basic_valid) / len(generated_df) * 100
        accuracy_fourier = len(valid_with_fourier) / len(generated_df) * 100
        print(f"Evaluation at Epoch {epoch}: Basic Accuracy: {accuracy_basic:.2f}%, Fourier Accuracy: {accuracy_fourier:.2f}%")
        
        
        eval_steps.append(epoch)
        basic_ratio_list.append(accuracy_basic)
        fourier_ratio_list.append(accuracy_fourier)
        
        
        if (accuracy_basic <= best_basic_ratio) or (accuracy_fourier <= best_fourier_ratio):
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}")
        else:
            no_improve_count = 0
            best_basic_ratio = max(best_basic_ratio, accuracy_basic)
            best_fourier_ratio = max(best_fourier_ratio, accuracy_fourier)
            best_generator_state = G_A2B.state_dict()
            torch.save(best_generator_state, 'G_A2B_best_model.pth')
            print("New best generator saved.")
        
        
        if no_improve_count >= patience:
            print("Early stopping: No improvement in 10 consecutive evaluations.")
            break


plt.figure(figsize=(10, 5))
plt.plot(eval_steps, basic_ratio_list, label='Basic QSC Accuracy')
plt.plot(eval_steps, fourier_ratio_list, label='Fourier QSC Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Evaluation Accuracy over Epochs')
plt.legend()
plt.savefig('accuracy_plots.png')
plt.show()
