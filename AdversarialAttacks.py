import numpy as np
import torch
from torch.autograd import Variable
from typing import List, Optional, Tuple
import torch.nn.functional as F
from tqdm import tqdm
import Levenshtein
import librosa
import scipy
from scipy import signal
import numpy as np

class ASRAttacks(object):
    '''
    Adversarial Attack on ASR model. Right now it is specifically implemented for 
    wav2vec2.0 model from torchaudio hub.

    It support the following attacks:
    1) Fast Gradient Sign Method  (FGSM)
    2) Basic Iterative Moment     (BIM)
    3) Projected Gradient Descent (PGD) 
    4) Carlini and Wagner Attack  (CW)
    5) Imperceptible ASR Attack   
    '''
    def __init__(self, model, device, labels: List[str]):
        '''
        Create an instance of the class "ASRAttacks"
        Input Arguments: 
        model  : The model on which the attack is supposed to be performed.
        device : Either 'cpu' if we have only CPU or 'cuda' if we have GPU
        labels : Label/Dictionary of the model.
        '''
        self.model = model
        self.device = device
        self.labels = labels

    def _encode_transcription(self, transcription: List[str]) -> List[str]: #in future add dictionary input over here
        '''
        Will encode transcription according to the dictionary of the model.
        Input Arguments:
        transcription    : transcription in a list. Ex: ["my name is mango"].
                           CAUTION:
                           Please make sure these characters are also present in the 
                           dictionary of the model also.
        '''
        # Define the dictionary
        dictionary = {'-': 0, '|': 1, 'E': 2, 'T': 3, 'A': 4, 
                      'O': 5, 'N': 6, 'I': 7, 'H': 8, 'S': 9, 
                      'R': 10, 'D': 11, 'L': 12, 'U': 13, 'M': 14, 
                      'W': 15, 'C': 16, 'F': 17, 'G': 18, 'Y': 19, 
                      'P': 20, 'B': 21, 'V': 22, 'K': 23, "'": 24, 
                      'X': 25, 'J': 26, 'Q': 27, 'Z': 28} #wav2vec uses this dictionary

        # Convert transcription string to list of characters
        chars = list(transcription)

        # Encode each character using the dictionary
        encoded_chars = [dictionary[char] for char in chars]

        # Concatenate the encoded characters to form the final encoded transcription
        encoded_transcription = torch.tensor(encoded_chars)

        # Returning the encoded transcription
        return encoded_transcription

    def FGSM_ATTACK(self, input_: torch.Tensor, target: List[str]= None, 
           epsilon: float = 0.2, targeted: bool = False) -> np.ndarray:

        '''
        Simple fast gradient sign method attack is implemented which is the simplest
        adversarial attack in this domain.
        For more information, see the paper: https://arxiv.org/pdf/1412.6572.pdf

        INPUT ARGUMENTS:

        input_        : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor

        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.

        epsilon       : Noise controlling parameter.
                        Type: float

        targeted      : If the attack should be targeted towards your desired 
                        transcription or not.
                        Type: bool
                        CAUTION:
                        Please make to pass your targetted 
                        transcription also in this case).

        RETURNS:

        np.ndarray : Perturbed audio
        '''
        # Making our input differentiable 
        input_.requires_grad = True

        # Forward Pass
        output, _ = self.model(input_.to(self.device))

        # Softmax Activation for computing logits
        output = F.log_softmax(output, dim=-1)

        if targeted:

            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)

            # Convert the target transcription to a PyTorch tensor
            target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

            # Computing CTC Loss
            output_lengths = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_lengths = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss = F.ctc_loss(output, target, output_lengths, target_lengths, blank=0, reduction='mean')

            # Zeroing the gradients of weights of the models
            self.model.zero_grad()

            # Computing gradient of our input w.r.t loss
            loss.backward()
            sign_grad = -input_.grad.data # If targeted then we will minimize our loss

            # Adding perturbation in the original input to make adversarial example    
            perturbation = epsilon * sign_grad
            perturbed_input = input_ + perturbation

            # Clamping the audio in original audio range (-1,1)
            perturbed_input = torch.clamp(perturbed_input, -1, 1)

            # returning perturbed audio
            return perturbed_input.detach().numpy()

        else:      

            # Using the model's transcription as target in untargeted attack
            untarget = self.INFER(input_.to(self.device))

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(untarget)

            # Convert the target transcription to a PyTorch tensor
            target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

            # Computing CTC Loss
            output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss = F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean')

            # Zeroing the gradients of weights of the models
            self.model.zero_grad()

            # Computing gradient of our input w.r.t loss
            loss.backward()

            # If untargeted then we will maximize our loss
            sign_grad = input_.grad.data 

            # Adding perturbation in the original input to make adversarial example    
            perturbed_input = input_ + (epsilon * sign_grad)

            # Clamping the audio in original audio range (-1,1)
            perturbed_input = torch.clamp(perturbed_input, -1, 1)

            # returning perturbed audio
            return perturbed_input.detach().numpy()

    def BIM_ATTACK(self, input_: torch.Tensor, target: List[str] = None,
          epsilon: float = 0.2, alpha: float = 0.1, 
          num_iter: int = 10, nested: bool = True,targeted: bool = False) -> np.ndarray:

        '''
        Basic Itertive Moment attack is implemented which is simple Fast Gradient 
        Sign Attack but in loop. 
        For more information, see the paper: https://arxiv.org/abs/1607.02533

        INPUT ARGUMENTS:

        input_        : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor

        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.

        epsilon       : Noise controlling parameter.
                        Type: float

        alpha         : Noise contribution in input controlling parameter
                        Type: float

        num_iter      : Number of iteration of attack
                        Type: int

        nested        : if this attack in being run in a for loop with tqdm 
                        Type: bool

        targeted      : If the attack should be targeted towards your desired 
                        transcription or not.
                        Type: bool
                        CAUTION:
                        Please make to pass your targetted 
                        transcription also in this case).

        RETURNS:

        np.ndarray : Perturbed audio
        ''' 
        # Making our input differentiable
        input_.requires_grad = True

        # Storing input in variable to add in noise later
        perturbed_input = input_

        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True

        if targeted:

            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)
            for _ in tqdm(range(num_iter), colour="red", leave = leave):

                output, _ = self.model(input_.to(self.device))

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Convert the target transcription to a PyTorch tensor
                target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

                # Zeroing the gradients of weights of the models
                self.model.zero_grad()

                # Computing CTC Loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean')

                # Computing gradients of our input w.r.t loss
                loss.backward()

                # If targeted then we will minimize our loss
                sign_grad = -input_.grad.data 

                # Adding perturbation in our input
                perturbed_input = perturbed_input + alpha * sign_grad

                # Clamping the perturbation in range (-eps, eps)
                eta = torch.clamp(perturbed_input - input_, -epsilon, epsilon)

                # Clamping the overall perturbated audio in the original audio range (-1, 1)
                perturbed_inputs = torch.clamp(input_ + eta, -1, 1)

            # returning perturbed audio after the loop ends
            return perturbed_inputs.detach().numpy()

        else:
            # Using the model's transcription as target in untargeted attack
            untarget = self.INFER(input_.to(self.device))

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(untarget)

            for _ in tqdm(range(num_iter), colour="red", leave = leave):
        
                output, _ = self.model(input_.to(self.device))

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Convert the target transcription to a PyTorch tensor
                target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

                # Zeroing the gradients of weights of the models
                self.model.zero_grad()

                # Computing CTC Loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean')

                # Computing gradients of our input w.r.t loss
                loss.backward()

                # If untargeted then we will maximize our loss
                sign_grad = input_.grad.data 

                # Adding perturbation in our input
                perturbed_input = perturbed_input + alpha * sign_grad

                # Clamping the perturbation in range (-eps, eps)
                eta = torch.clamp(perturbed_input - input_, -epsilon, epsilon)

                # Clamping the overall perturbated audio in the original audio range (-1, 1)
                perturbed_inputs = torch.clamp(input_ + eta, -1, 1)

            # returning perturbed audio after the loop ends
            return perturbed_inputs.detach().numpy()

    def PGD_ATTACK(self, input_: torch.Tensor, target: List[str] = None,
                 epsilon: float = 0.3, alpha: float = 0.01, num_iter: int = 40,
                 nested: bool = True,targeted: bool = False) -> np.ndarray:

        '''
        Projected Gradient Descent attack is implemented which in simple terms is more 
        advanced version of BIM. In this attack we project our perturbation back to 
        some Lp norm and find perturbations in that particular region. 
        For more information, see the paper: https://arxiv.org/abs/1706.06083

        INPUT ARGUMENTS:

        input_        : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor

        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.

        epsilon       : Noise controlling parameter.
                        Type: float

        alpha         : Noise contribution in input controlling parameter
                        Type: float

        num_iter      : Number of iteration of attack
                        Type: int

        nested        : if this attack in being run in a for loop with tqdm 
                        Type: bool

        targeted      : If the attack should be targeted towards your desired 
                        transcription or not.
                        Type: bool
                        CAUTION:
                        Please make to pass your targetted 
                        transcription also in this case).

        RETURNS:

        np.ndarray : Perturbed audio
        ''' 

        # Making our input differentiable
        input_audio = input_.clone().detach().requires_grad_(True)

        # Putting our model in eval mode
        self.model.eval()

        # Storing the initial audio
        original_audio = input_.clone().detach()

        if targeted: 
            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)

        else: # If untargeted
            untarget = self.INFER(input_audio) # Then we will use the original input's transcription as our target

            # Encoding the untargeted transcription
            encoded_transcription = self._encode_transcription(untarget)

            # Making this value true because now we have target to compute loss
            target = True

        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True

        for _ in tqdm(range(num_iter), colour = 'red', leave = leave):

            # Calculate the CTC loss and gradients
            input_audio.requires_grad = True

            # Forward pass
            output, _ = self.model(input_audio.to(self.device))

            # Softmax Activation for computing logits
            output = F.log_softmax(output, dim=-1)

            if target is not None:

                # Preparing arguments for computing CTC Loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

            # Computing CTC loss
            loss = F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean', zero_infinity=True) if targeted else -F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean', zero_infinity=True)

            # Computing gradients of our input w.r.t loss
            loss.backward()

            # Update the audio with the calculated gradients
            with torch.no_grad():

                    # Calculating perturbation to be added in input audio
                    perturbation = torch.sign(input_audio.grad.data) * alpha

                    # Adding calculated perturbation in the input audio
                    input_audio += perturbation

                    # Clip the audio within the epsilon boundary
                    delta = torch.clamp(input_audio - original_audio, min=-epsilon, max=epsilon)

                    # Clamping the overall perturbed audio in between the audio range
                    input_audio = (original_audio + delta).clamp(min=-1, max=1)
    
        #returning perturbed adversarial example
        return input_audio.detach().numpy()

    def CW_ATTACK(self, input_: torch.Tensor, target: List[str] = None,
           epsilon: float = 0.3, c: float = 1e-4, learning_rate: float = 0.01,
           num_iter: int = 1000, decrease_factor_eps: float = 0.8,
           num_iter_decrease_eps: int = 10, optimizer: str = None,
           threshold: float = 0.5, nested: bool = True, targeted: bool = False) -> np.ndarray:

        '''
        Implements the Carlini and Wagner attack, the strongest white box 
        adversarial attack. This attack uses an optimization strategy to find the 
        adversarial transcription while keeping the perturbation as low as possible. 
        For more information, see the paper: https://arxiv.org/pdf/1801.01944.pdf

        INPUT ARGUMENTS:

        input_        : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor

        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.

        epsilon       : Noise controlling parameter.
                        Type: float

        c             : Regularization term controlling factor.
                        Type: float

        learning_rate : learning_rate of optimizer.
                        Type: float

        num_iter      : Number of iteration of attack.
                        Type: int

        decrease_factor_eps   : Factor to decrease epsilon by during optimization 
                                Type: float

        num_iter_decrease_eps : Number of iterations after which to decrease epsilon
                                Type: int

        optimizer     : Name of the optimizer to use for the attack. 
                        Type: str

        threshold     : threshold for lowering epsilon value
                        Type: float

        nested        : if this attack in being run in a for loop with tqdm 
                        Type: bool

        targeted      : If the attack should be targeted towards your desired 
                        transcription or not.
                        Type: bool
                        CAUTION:
                        Please make to pass your targetted 
                        transcription also in this case).
        RETURNS:

        np.ndarray : Perturbed audio
        '''

        # Convert the input audio to a PyTorch tensor
        input_audio = input_.to(self.device).float()

        # Making audio differentiable
        input_audio.requires_grad_()

        # Cloning the original audio 
        input_audio_orig = input_.clone().to(self.device)

        if targeted:

            # Making sure target is given in targeted attack
            assert target is not None, "Target should not be 'None' in a targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)

        else: # If untargeted

            # Then we will use the model's transcription as our target
            untarget = self.INFER(input_audio.to(self.device)) 

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(untarget)

        # Convert the target transcription to a PyTorch tensor
        target_tensor = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

        if optimizer == "Adam":

            # Define the optimizer
            optimizer = torch.optim.Adam([input_audio], lr=learning_rate)

        else:

            # Define the optimizer
            optimizer = torch.optim.SGD([input_audio], lr=learning_rate)

        # Setting our inital parameters
        successful_attack = False 
        num_successful_attacks = 0

        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True

        for i in tqdm(range(num_iter), colour="red", leave = leave):

            # Zero the gradients
            optimizer.zero_grad()

            # Compute the model’s prediction
            output, _ = self.model(input_audio)

            # Softmax Activation for computing logits
            output = F.log_softmax(output, dim=-1)

            # Compute the CTC loss function
            output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss_classifier = F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean', zero_infinity=True) if targeted else -F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean', zero_infinity=True)

            # Regularization term to minimize the perturbation
            loss_regularizer = c * torch.norm(input_audio - input_audio_orig)

            # Combine the losses and compute gradients
            loss = loss_classifier + loss_regularizer

            # Computing gradients of our input w.r.t loss
            loss.backward()

            # Update the input audio with gradients
            optimizer.step()

            # Calculating perturbation by subtracting the optimized audio from cloned one
            perturbation = input_audio - input_audio_orig

            # Project the perturbation onto the epsilon ball in range (-eps, eps)
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)

            # Cliping to audio in range (-1, 1)
            input_audio.data = torch.clamp(input_audio_orig + perturbation, -1, 1)

            # Check if the attack is successful or not using Levenshtein distance
            predicted_transcription = self.INFER(input_audio)
            if Levenshtein.ratio(predicted_transcription, target) > threshold:
                print(Levenshtein.ratio(predicted_transcription, target))
                print("This is predicted: ",list(predicted_transcription))
                print("This is our target: ",target)
                num_successful_attacks += 1
                if num_successful_attacks >= num_iter_decrease_eps:
                    successful_attack = True
                    epsilon *= decrease_factor_eps
                    print("Epsilon value after decrease: ", epsilon)
                    num_successful_attacks = 0
                else:
                    successful_attack = False
                    
            if successful_attack and epsilon <= 0:
                break

        return input_audio.detach().cpu().numpy()

    def IMPERCEPTIBLE_ATTACK(self, input_: torch.Tensor, target: List[str] = None,
                             epsilon: float = 0.3, c: float = 1e-4, learning_rate1: float = 0.01, 
                             learning_rate2: float = 0.01, num_iter1: int = 10000, num_iter2: int = 2000, 
                             decrease_factor_eps: float = 0.8, num_iter_decrease_eps: int = 10, 
                             optimizer1: str = None, optimizer2: str = None, threshold: float = 0.5, 
                             nested: bool = True , alpha: float = 0.5) -> np.ndarray:
        
        '''
        Implements the Imperceptible ASR attack, which leverages the strongest white box 
        adversarial attack which is CW attack while also masking sure the added perturbation
        is imperceptible to humans using Psychoacoustic Scale. This attack is performed in two 
        stages. In first stage we perform simple CW attack and in 2nd stage we make sure our 
        added perturbations are imperceptible. 
        For more information, see the paper: https://arxiv.org/abs/1903.10346

        INPUT ARGUMENTS:

        input_        : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor

        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.

        epsilon       : Noise controlling parameter.
                        Type: float

        c             : Regularization term controlling factor.
                        Type: float

        learning_rate1: learning_rate of optimizer for stage 1.
                        Type: float
        
        learning_rate2: learning_rate of optimizer for stage 2.
                        Type: float

        num_iter1     : Number of iteration of attack stage 1.
                        Type: int
        
        num_iter2     : Number of iteration of attack stage 2.
                        Type: int

        decrease_factor_eps   : Factor to decrease epsilon by during optimization 
                                Type: float

        num_iter_decrease_eps : Number of iterations after which to decrease epsilon
                                Type: int

        optimizer1     : Name of the optimizer to use for the attack stage 1. 
                         Type: str

        optimizer2     : Name of the optimizer to use for the attack stage 2. 
                         Type: str

        nested         : if this attack in being run in a for loop with tqdm 
                        Type: bool
        
        threshold      : threshold for lowering epsilon value
                         Type: float

        alpha          : controlling factor for second stage loss.
                         Type: float

        RETURNS:

        np.ndarray : Perturbed audio
        '''
        
        # This attack will be targeted for now, therefore...
        assert (target is not None), "Please pass a specific target transcription for performing this attack"

        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True
        
        print("*"*5,"Attack Stage 1","*"*5) #stage 1 of Imperceptible ASR attack
        stageOneAud =  self.CW_ATTACK(input_ , target = target, epsilon = epsilon, c = c, learning_rate = learning_rate1,
                           num_iter = num_iter1, decrease_factor_eps = decrease_factor_eps, 
                           num_iter_decrease_eps = num_iter_decrease_eps, optimizer = optimizer1, 
                           threshold = threshold, nested = True, targeted = True)
        
        # Convert the input audio to a PyTorch tensor
        input_audio = torch.from_numpy(stageOneAud).to(self.device).float()

        # Making audio differentiable
        input_audio.requires_grad_()

        # Cloning the original audio 
        input_audio_orig = input_.clone().to(self.device)
        
        # PSD and threshold calculation
        theta, original_max_psd = self._compute_masking_threshold(input_.numpy().squeeze(), win_length = 2048, hop_length = 512, 
                                                                 n_fft = 2048, sample_rate = 16000)
        
        # Taking transpose of threshold to store it in correct order
        theta = theta.transpose(1, 0)
          
        if optimizer2 == "Adam":

            # Define the optimizer
            optimizer = torch.optim.Adam([input_audio], lr=learning_rate2)

        else:

            # Define the optimizer
            optimizer = torch.optim.SGD([input_audio], lr=learning_rate2)
        
        # Encode the target transcription
        encoded_transcription = self._encode_transcription(target)
        
        # Convert the target transcription to a PyTorch tensor
        target_tensor = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()
        
        # Preparing relu activation for furthur use
        relu = torch.nn.ReLU()
        
        print("*"*5,"Attack Stage 2","*"*5) #stage 2 of Imperceptible ASR attack
        for i in tqdm(range(num_iter2), colour = 'red', leave = leave):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Compute the model’s prediction
            output, _ = self.model(input_audio)

            # Softmax Activation for computing logits
            output = F.log_softmax(output, dim=-1)

            # Compute the CTC loss function
            output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss_classifier = F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean', zero_infinity=True)

            # Regularization term to minimize the perturbation
            loss_regularizer = c * torch.norm(torch.from_numpy(stageOneAud).to(self.device) - input_audio_orig)

            # Combine the losses and compute gradients
            loss1 = loss_classifier + loss_regularizer
            
            # Calculating perturbation by subtracting the optimized audio from cloned one
            perturbation = input_audio - input_audio_orig
            
            # Calculating PSD matrix of perturbation which is added in the clean audio
            psd_transform_delta = self._psd_transform(
                delta=perturbation, original_max_psd=original_max_psd
            )
            
            # Calculating the loss between perturbation's PSD and original audio's PSD
            loss2 = torch.mean(relu(psd_transform_delta - torch.tensor(theta).to(self.device)))
            
            # Summing both CW loss and PSD loss
            loss = loss1.type(torch.float64) + torch.tensor(alpha).to(self.device) * loss2
            
            # Taking mean of both losses
            loss = torch.mean(loss)
            
            # Computing gradients of our input w.r.t loss
            loss.backward()

            # Update the input audio with gradients
            optimizer.step()

        return input_audio.detach().cpu().numpy() 
            
        
    def _psd_transform(self, delta: "torch.Tensor", original_max_psd: np.ndarray) -> "torch.Tensor":
        
        '''
        Note:
        This code is taken from ART Toolbox by IBM.
        
        Computes the PSD matrix of the perturbation.
        
        INPUT ARGUMENTS:
        
        delta            : It is the perturbation added in the audio.
                           Type: torch.Tensor
        
        original_max_psd : It is the maximum PDF of the original clean audio.
                           Type: np.ndarray
                           
        RETURNS:

        torch.Tensor : The psd matrix.
        '''
        
        import torch

        # Get window for the transformation
        window_fn = torch.hann_window  # type: ignore

        # Return STFT of delta
        delta_stft = torch.stft(
          delta,
          n_fft=2048,
          hop_length=512,
          win_length=2048,
          center=False,
          window=window_fn(2048).to(self.device),
        ).to(self.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))

        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / 2048
        psd = psd ** 2
        psd = (
          torch.pow(torch.tensor(10.0).type(torch.float64), torch.tensor(9.6).type(torch.float64)).to(
              self.device
          )
          / torch.reshape(torch.tensor(original_max_psd).to(self.device), [-1, 1, 1])
          * psd.type(torch.float64)
        )

        return psd
        
    def _compute_masking_threshold(self, x: np.ndarray, win_length: int = 2048,
                                  hop_length: int = 512, n_fft: int = 2048, 
                                  sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        
        '''
        Note:
        This code is taken from ART Toolbox by IBM.
        
        Computes the masking threshold and the maximum psd of the original audio.
        
        INPUT ARGUMENTS:
        
        delta            : Original clean audio of shape (seq_length,)
                           Type: np.ndarray
        
        win_length       : Window length of STFT.
                           Type: int
                           
        hop_length       : hop length of STFT.
                           Type: int
                    
        n_fft            : fft-points argument of STFT.
                           Type: int
    
        sample_rate      : sample rate of original audio
                           Type: int
                           
        RETURNS:

        Tuple[np.ndarray, np.ndarray] : A tuple containing (masking threshold, maximum psd)
        '''
        
        # First compute the psd matrix
        # Get window for the transformation
        window = scipy.signal.get_window("hann", win_length, fftbins=True)

        # Do transformation
        transformed_x = librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
        transformed_x *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_x / win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * np.log10(psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        freqs = librosa.core.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float64) - np.inf
        bark_idx = int(np.argmax(barks > 1))
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []

            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = int(np.argmax(d_z > 0))
                s_f = np.zeros(len(d_z), dtype=np.float64)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s_array = np.array(t_s)

            theta.append(np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0))

        theta_array = np.array(theta)

        return theta_array, original_max_psd
    
    def INFER(self, input_: torch.Tensor) -> str:
        
        '''
        Method for performing inference by the model.
        
        INPUT ARGUMENTS:
        
        input_        : Input audio of shape Ex: [0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                         
        RETURNS:

        str           : Model's transcription from the given audio.
        '''

        # Inference method of the model
        blank = 0
        output, _ = self.model(input_.to(self.device))
        encodedTrans = torch.argmax(output[0], axis=-1)
        encodedTrans = torch.unique_consecutive(encodedTrans, dim=-1)
        indices = [i for i in encodedTrans if i != blank]
        return "".join([self.labels[i] for i in indices])
    