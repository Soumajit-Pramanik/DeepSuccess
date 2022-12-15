import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import os
import math

from data import get_grp_cat, get_metric_ch, loadfiles, get_thres

MAX_INP_LEN = 5
MAX_LENGTH = 10


def data2Tensor(train_list, batch_size, device):
    input_var = []
    target_var = []
    gt = []
    grp = []
    citt = []
    tag = []
    for i, d in enumerate(train_list):
        #print(d[0])
        input_var.append(d[0])
        target_var.append(d[1])
        gt.append(d[2])
        grp.append(d[3])
        citt.append(d[4])
        tag.append(d[5])
        #print(input_var, output_var)
        #print(input_var.shape, output_var.shape)
    input_var = torch.FloatTensor(input_var)
    target_var = torch.FloatTensor(target_var)

    temp = np.ones(batch_size)
    for i, d in enumerate(temp):
        temp[i] = MAX_INP_LEN
        
    lengths = torch.FloatTensor(temp)
    lengths = lengths.to(device)
    input_var = input_var.transpose(0,1)
    input_var = input_var.to(device)
    target_var = target_var.transpose(0,1)
    target_var = target_var.to(device)

    return input_var, lengths, target_var, [gt,grp,citt,tag]

def valtovec(f_out, f_out2, others, device):
    bint = np.linspace(0,1,6)
    real_out = []
    for i, _ in enumerate(f_out):
        tfa = []
        tfb = []
        for j in range(5):
            onea = [0,0,0,0,0]
            oneb = [0 for _ in range(5)]

            for n in range(0, len(bint)-1):
                b_start = bint[n]
                b_end = bint[n+1]
                if float(f_out[i][j])>=b_start and float(f_out[i][j])<b_end:
                    onea[n]=1
                if float(f_out2[i][j])>=b_start and float(f_out2[i][j])<b_end:
                    oneb[n]=1

            tfa.extend(onea)
            tfb.extend(oneb)
        tfa.extend(tfb)
        tfa.extend(int(d) for d in others[3][i])
        real_out.append(tfa)

    real_out = torch.FloatTensor(real_out)
    real_out = real_out.to(device)
    return real_out


def SELoss(output, target):
    # target = target.view(-1,1)
    # # decoder-out shape: (batch_size, vocab_size) , target_size = (batch_size, 1)
    # target = target.type(torch.LongTensor)
    # gathered_tensor = torch.gather(output, 1, target)
    # # Calculate the Negative Log Likelihood Loss
    # crossEntropy = -torch.log(gathered_tensor)
    # # Calculate the eman of the loss
    # loss = crossEntropy.mean()
    # loss = loss.to(device)
    target = target.view(-1,1)
    output = output.view(-1,1)
    #loss = torch.mean((target.squeeze(1) - output.squeeze(1))**2)
    criterion = nn.MSELoss()
    loss = criterion(target,output)
    return loss, output.shape[0]


def pAccuracy(others, f_out, f_tar, metadata, gs):
    #print("Testing Loss = ", print_loss)
    f_out = np.array(f_out)
    f_tar = np.array(f_tar)
    #print("ffout_shape=",f_out.shape,"  fftar_shape=",f_tar.shape)
    gt, grp, citt, _ = others
    gt = np.array(gt)
    grp = np.array(grp)
    citt = np.array(citt)

    grp_cat_array = loadfiles()
    _, maxval, thres1, thres2 = metadata
    
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(f_out.shape[0]):
        citv=int(citt[i])
        if get_metric_ch(grp[i],citv,grp_cat_array) == (2 if gs else 1):
            gcatt=get_grp_cat(grp[i],citv,grp_cat_array)
            if gs:
                e_succ_thres=thres2[citv][gcatt][1]
                e_fail_thres=thres1[citv][gcatt][1]
                pred2=int((sum(f_out[i][0] for j in range(5))*maxval[citv])/5)
            else:
                e_succ_thres=thres2[citv][gcatt][0]/1000
                e_fail_thres=thres1[citv][gcatt][0]/1000
                #print pred,act,e_succ_thres,e_fail_thres
                pred2=int(sum((f_out[i][j]-f_out[i][j-1])/f_out[i][j-1] for j in range(1,5))*maxval[citv]/5)
            if pred2 >= e_succ_thres and gt[i] == 1:
                tp+=1
            if pred2 >= e_succ_thres and gt[i] == 0:
                fp+=1
            if pred2 <= e_fail_thres and gt[i] == 1:
                fn+=1
            if pred2 <= e_fail_thres and gt[i] == 0:
                tn+=1

    if (tp+tn+fp+fn) > 0:
        acc = float(tp+tn)/(tp+tn+fp+fn)
        #print('Accuracy = ',acc)
    #else:
        #print('All are Zero')

    #print(tp,tn,fp,fn)

    if (tp + fp) > 0:
        prec = float(tp)/(tp+fp)
        #print('Precision = ',prec) 
    if (tp + fn) > 0:
        rec = float(tp)/(tp+fn)
        #print('Recall = ',rec)
    if( (tp+fp)>0 and (tp+fn)>0 ):
        f1s = (2*rec*prec)/(rec+prec)
        #print("F1-score = {:.4f}".format(f1s))


def pAccuracy_bin(f_out, f_tar, bin_sizes):
    f_out = np.array(f_out)
    f_tar = np.array(f_tar)

    if(f_out.shape[0] == 0):
        print("pAccuracy_bin : size of input 0")
        return 
    #print("Events - 6 7 8 9 10 6and7 6-10")
    print("Binning Accuracy: ")
    for bin_size in bin_sizes:
        bint = np.linspace(0,1,bin_size+1)
        #print("For bin size ", bin_size," :",end=" ")
        for i in range(5):
            ae = []
            tz = oz = 0
            for itr in range(f_out.shape[0]):
                onep=[0,0,0,0,0,0,0,0,0,0]
                oner=[0,0,0,0,0,0,0,0,0,0]

                for n in range(0,len(bint)-1):
                    b_start=bint[n]
                    b_end=bint[n+1]
                    if(f_tar[itr][i] < 0):
                        tz+=1
                    if(f_out[itr][i] < 0):
                        oz+=1
                    if float(f_tar[itr][i])>=b_start and float(f_tar[itr][i])<=b_end:
                        oner[n]=1
                    if float(f_out[itr][i])>=b_start and float(f_out[itr][i])<=b_end:
                        onep[n]=1

                if int(1) not in onep or int(1) not in oner:
                    continue
                diff=onep.index(1)-oner.index(1)
                if abs(diff)==0:
                    ae.append(1)
                else:
                    ae.append(0)

            if(len(ae)==0):
                #print("\tEvent_Number-",(i+6)," All negative", end='')
                print("0")#,end = ' ')
            else:
                mae=float(sum(ae))/float(len(ae))
                # print("\tEvent_Number-",(i+6)," MAE",mae, end='')
                #print("{:.4f}".format(mae),end=" ")
            #print("\t\t excluded(negative) :", oz)

        se = []
        for itr in range(f_out.shape[0]):
            onep67=[0,0,0,0,0,0,0,0,0,0]
            oner67=[0,0,0,0,0,0,0,0,0,0]
            a67=(float(f_tar[itr][0])+float(f_tar[itr][1]))*0.5
            p67=(float(f_out[itr][0])+float(f_out[itr][1]))*0.5
            for n in range(0,len(bint)-1):
                b_start=bint[n]
                b_end=bint[n+1]
                if a67>=b_start and a67<=b_end:
                    oner67[n]=1
                if p67>=b_start and p67<=b_end:
                    onep67[n]=1
            if int(1) not in onep67 or int(1) not in oner67:
                continue
            diff1=onep67.index(1)-oner67.index(1)
            if abs(diff1)==0:
                se.append(1)
            else:
                se.append(0)
        if(len(se)):
            mse=float(sum(se))/float(len(se))
            # print("\tEvent_Number- 6and7"," MSE",mse)
            #print("{:.4f}".format(mse),end=" ")
        else:
            # print("\tEvent_Number- 6and7"," All negative")
            print("0")#,end = ' ')

        pe = []
        for itr in range(f_out.shape[0]):
            onep60=[0,0,0,0,0,0,0,0,0,0]
            oner60=[0,0,0,0,0,0,0,0,0,0]
            a60= (sum(float(f_tar[itr][j]) for j in range(5)))*0.2
            p60= (sum(float(f_out[itr][j]) for j in range(5)))*0.2
            for n in range(0,len(bint)-1):
                b_start=bint[n]
                b_end=bint[n+1]
                if a60>=b_start and a60<=b_end:
                    oner60[n]=1
                if p60>=b_start and p60<=b_end:
                    onep60[n]=1
            if int(1) not in onep60 or int(1) not in oner60:
                continue
            diff2=onep60.index(1)-oner60.index(1)
            if abs(diff2)==0:
                pe.append(1)
            else:
                pe.append(0)
        
        if(len(pe)):
            mpe=float(sum(pe))/float(len(pe))
            # print("\tEvent_Number- 6-10"," MPE",mpe)
            print("{:.4f}".format(mpe))
        else:
            # print("\tEvent_Number- 6-10"," All negative")
            print("0")



def train(input_variable, lengths, target_variable, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, tf_ratio, max_length=MAX_LENGTH, device="cpu"):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.FloatTensor([[[0] for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < tf_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    f_output, f_target = [], []
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1, 1)
            # Calculate and accumulate loss
            tloss, nTotal = SELoss(decoder_output, target_variable[t])
            loss += tloss
            print_losses.append(tloss.item() * nTotal)
            n_totals += nTotal

            f_output.append(decoder_output.view(-1).cpu().detach().numpy())
            f_target.append(target_variable[t].view(-1).cpu().detach().numpy())

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            #_, topi = decoder_output.topk(1)
            decoder_input = decoder_output.view(1, -1, 1)
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            tloss, nTotal = SELoss(decoder_output, target_variable[t])
            loss += tloss
            print_losses.append(tloss.item() * nTotal)
            n_totals += nTotal

            f_output.append(decoder_output.view(-1).cpu().detach().numpy())
            f_target.append(target_variable[t].view(-1).cpu().detach().numpy())

    f_out = np.transpose(np.array(f_output))
    f_tar = np.transpose(np.array(f_target))
    
    return loss, sum(print_losses) / n_totals, f_out, f_tar


def trainf(fmodel, inp_vec, other, device):
    
    gt, _, _, _ = other
    gt = torch.FloatTensor(gt)
    gt = gt.to(device)

    output = fmodel(inp_vec)
    tloss, nt = SELoss(output, gt)

    return tloss, tloss/nt


def trainItersCombined(train_list, train_list2, metadata, metadata2, model_name, hidden_size, encoder, decoder,
                 encoder2, decoder2, fmodel, lmodel, encoder_optimizer, decoder_optimizer,
                 encoder_optimizer2, decoder_optimizer2, fmodel_optimizer, lmodel_optimizer,
                 encoder_n_layers, decoder_n_layers, max_target_len, save_dir, n_iteration,
                 batch_size, print_every, save_every, clip, tf_ratio, loadFilename, device, checkpoint=0):

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    start = 0
    tr_size = min(len(train_list),len(train_list2))

    # Training loop
    print("Starting Training Iterations...\n")
    for iteration in range(start_iteration, n_iteration + 1):
        if((start+batch_size)>tr_size):
            start = 0
        training_batches = train_list[start:start+batch_size]
        # Extract fields from batch
        input_variable, lengths, target_variable, other = data2Tensor(training_batches, batch_size, device)

        training_batches2 = train_list2[start:start+batch_size]
        # Extract fields from batch
        input_variable2, lengths2, target_variable2, other2 = data2Tensor(training_batches2, batch_size, device)

        # Scheduled sampling implementation
        tf_ratio = math.exp( -1 * ( (iteration-1)/n_iteration ) * 5 )

        # Run a training iteration with batch
        loss , ploss, f_out, f_tar = train(input_variable, lengths, target_variable, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, tf_ratio, device=device)

        loss2, ploss2, f_out2, f_tar2 = train(input_variable2, lengths2, target_variable2, max_target_len, encoder2,
                     decoder2, encoder_optimizer2, decoder_optimizer2, batch_size, clip, tf_ratio, device=device)
        
        # inp_vec = valtovec(f_out, f_out2, other, device)
        # print('-----------------------------------------"""DEVICE"""---------------------------------------------')
        # print(device)

        f_out3 = []
        for i in range(len(f_out)):
            temp = []
            temp.extend(f_out[i])
            temp.extend(f_out2[i])
            temp.extend(int(d) for d in other[3][i])
            f_out3.append(temp)
        f_out3 = torch.FloatTensor(f_out3)
        f_out3.to(device)

        # f_out3 = torch.cat((f_out,f_out2),1)
        loss3, ploss3 = trainf(fmodel, f_out3, other, device)

        floss = loss + loss2 + loss3

        # iloss = []
        # iloss.append([loss,loss2,loss3])
        # iloss = torch.FloatTensor(iloss)
        # iloss.to(device)

        # iloss = torch.cat((loss, loss2, loss3), 1)
        # floss = lmodel(iloss)

        # Perform backpropatation
        floss.backward()
        
        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(encoder2.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder2.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(fmodel.parameters(), clip)
        #_ = torch.nn.utils.clip_grad_norm_(lmodel.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer2.step()
        decoder_optimizer2.step()
        fmodel_optimizer.step()
        #lmodel_optimizer.step()
        
        ptloss = ploss + ploss2 + ploss3
        print_loss += (ptloss)

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Average loss: {:.6f}".format(iteration, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

    print("----------Training Done-------------")

def eval_data(test_list, metadata, encoder, decoder, encoder_n_layers, decoder_n_layers, max_target_len, batch_size, device):

    #print("Initializing testing.......")
    start = 0
    loss = 0
    print_losses = []
    n_totals = 0
    print_loss = 0
    itr = 0

    f_out = []
    f_tar = []
    others = []
    others.extend([] for _ in range(4))
    tsl = len(test_list)
    #print("Data size to be evaluated =",tsl)

    with torch.no_grad():
        while(start< int(tsl - int(batch_size))):
            # Initialize variables
            loss = 0
            print_losses = []
            n_totals = 0

            testing_batch = test_list[start:start+batch_size]
            start += batch_size

            input_variable, lengths, target_variable, other = data2Tensor(testing_batch, batch_size, device)
            # Forward pass through encoder
            encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.FloatTensor([[[0] for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]

            f_output, f_target = [], []
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = target_variable[t].view(1, -1, 1)

                tloss, nTotal = SELoss(decoder_output, target_variable[t])
                loss += tloss
                print_losses.append(tloss.item() * nTotal)
                n_totals += nTotal

                #print("dec_out_shape=",decoder_output.shape, " target_var_shape=",target_variable[t].shape )
                f_output.append(decoder_output.view(-1).cpu().numpy())
                f_target.append(target_variable[t].view(-1).cpu().numpy())
                #print("new_dec_shape=",decoder_output.detach().numpy().shape)

            print_loss+=(sum(print_losses)/n_totals)

            itr+=1
            if(itr%20 == 0):
                print_loss_avg = print_loss / (20*batch_size)
                print("eval data: {}; Percent complete: {:.1f}%; Average loss: {:.6f}".format(start, start / tsl * 100, print_loss_avg))
                print_loss = 0

            f_output = np.array(f_output)
            f_target = np.array(f_target)
            #print("f_out_shape=",f_output.shape, " f_tar_shape=",f_target.shape)
            f_out.extend(np.transpose(f_output))
            f_tar.extend(np.transpose(f_target))
            for i in range(4):
                others[i].extend(other[i])
    
    return others, f_out, f_tar
    # print("++++++ Testing Accuracy ++++++")
    # pAccuracy(others, f_out, f_tar, metadata, gs)
    # pAccuracygs_bin(f_out, f_tar)


def pcAccuracy(fmodel, f_out, f_out2, other, device, thres=0):
    inp_vec = []
    for i in range(len(f_out)):
        temp = []
        temp.extend(f_out[i])
        temp.extend(f_out2[i])
        temp.extend(int(d) for d in other[3][i])
        inp_vec.append(temp)
    inp_vec = torch.FloatTensor(inp_vec)
    inp_vec.to(device)
    # inp_vec = valtovec(f_out, f_out2, other, device)
    
    # print("pcAccuracy : size of inp_vec : ",inp_vec.shape)
    with torch.no_grad():
        gt, _, _, _ = other

        out = []
        # for i in range(inp_vec.shape[0]):
        output = fmodel(inp_vec)
            # if(output>thres):
            #     out.append(1)
            # else:
            #     out.append(0)
        output = output.tolist()
        # if thres is 0:
        maxv = max([d[0] for d in output])
        minv = min([d[0] for d in output])
        
        outs = []
        thress = []
        for j in range(21):
            thres = minv + (j/20)*(maxv-minv)
            out = [0 if d[0]<thres else 1 for d in output]
            outs.append(out)
            thress.append(thres)

        # print("pcAccuracy : size of gt : ",len(gt))
        # print("pcAccuracy : size of out : ", len(out))
        #print(out)

        macc = 0
        for d, out in enumerate(outs):
            
            
            #print("Thres = ",thress[d],end="   ")
            tp=0
            fp=0
            tn=0
            fn=0

            for i in range(len(out)):
                if out[i]==1 and gt[i]==1:
                    tp+=1
                if out[i]==1 and gt[i]==0:
                    fp+=1
                if out[i]==0 and gt[i]==1:
                    fn+=1
                if out[i]==0 and gt[i]==0:
                    tn+=1

            if (tp+tn+fp+fn) > 0:
                acc = float(tp+tn)/(tp+tn+fp+fn)
                # print("Accuracy = {:.4f}".format(acc),end="   ")
            #else:
                # print('All are Zero')

            # print(tp,tn,fp,fn,end='  ')

            if (tp + fp) > 0:
                prec = float(tp)/(tp+fp)
                # print("Precision = {:.4f}".format(prec),end="   ") 
            if (tp + fn) > 0:
                rec = float(tp)/(tp+fn)
                # print("Recall = {:.4f}".format(rec))
            if( (rec+ prec)>0):
                f1 = (2*rec*prec)/(rec+prec)
                # print("F1-score = {:.4f}".format(f1s))
            if(acc > macc):
                mf1 = f1
                macc = acc
                mprec = prec
                mrec = rec

        print("Accuracy = {:.4f}".format(macc))
        print("Precision = {:.4f}".format(mprec))
        print("Recall = {:.4f}".format(mrec))
        print("F1-score = {:.4f}".format(mf1))
