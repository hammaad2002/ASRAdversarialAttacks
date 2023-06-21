import torch.nn.functional as F
from typing import List, Tuple
from tqdm.notebook import tqdm
from scipy import signal
import librosa
import numpy as np
import torch
import scipy
from functools import reduce
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer

class ASRAttacks(object):
    '''
    Adversarial Attack on ASR model. Right now it is specifically implemented for 
    wav2vec2.0 model from torchaudio hub.
    
    It support the following attacks:
    1) Fast Gradient Sign Method  (FGSM)
    2) Basic Iterative Moment     (BIM)
    3) Projected Gradient Descent (PGD) 
    4) Carlini and Wagner Attack  (CW)
    5) Imperceptible ASR Attack   (IMP-ASR)
    '''
    def __init__(self, model: str, device):
        '''
        Creates an instance of the class "ASRAttacks".
        
        INPUT ARGUMENTS: 
        
        model  : Model's name which is shown on huggingface's website
        device : Either 'cpu' if we have only CPU or 'cuda' if we have GPU
        '''
        self.model = AutoModelForCTC.from_pretrained(model).to(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device

    def _encode_transcription(self, transcription: List[str]) -> List[str]: #in future add dictionary input over here
        '''
        Will encode transcription according to the dictionary of the model.
        
        INPUT ARGUMENTS:
        
        transcription    : transcription in a list. Ex: ["my name is mango"].
                           CAUTION:
                           Please make sure these characters are also present in the 
                           dictionary of the model also.
        '''
        # Making sure the transcription matches our model's token also
        if transcription.isupper():
            transcription = transcription
        elif transcription .islower():
            transcription = transcription.upper()
        else: # useless condition
            transcription = transcription.upper()

        # Encoding our transcription with the corresponding tokens
        encoded_transcription = self.tokenizer.encode(transcription, return_tensors ="pt")

        # Returning the encoded transcription
        return encoded_transcription

    def FGSM_ATTACK(self, input__: torch.Tensor, target: List[str]= None, 
           epsilon: float = 0.2, targeted: bool = False) -> np.ndarray:
        
        '''
        Simple fast gradient sign method attack is implemented which is the simplest
        adversarial attack in this domain.
        For more information, see the paper: https://arxiv.org/pdf/1412.6572.pdf
        
        INPUT ARGUMENTS:
        
        input__       : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                        
        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.
                        
        epsilon       : Noise controlling parameter or step size.
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
        # Cloning the original audio
        input_ = input__.clone()
        
        # Making our input differentiable 
        input_.requires_grad = True

        # Forward Pass
        output = self.model(input_.to(self.device)).logits

        # Softmax Activation for computing logits
        output = F.log_softmax(output, dim=-1)

        if targeted: # Condition for checking if the user wants 'targeted' attack to be performed

            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."
            
            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)

            # Convert the target transcription to a PyTorch tensor
            target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

            # Computing the CTC Loss
            output_lengths = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_lengths = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss = F.ctc_loss(output, target, output_lengths, target_lengths, blank=0, reduction='mean')

            # Computing gradient of our input w.r.t loss
            loss.backward()
            
            # If 'targeted' then we will minimize our loss to the respective target transcription
            sign_grad = -input_.grad.sign() 

            # Calculating 'sign' of the FGSM attack and multiplying it with our small epsilon step
            perturbation = epsilon * sign_grad
            
            # Adding perturbation in the original input to make adversarial example   
            perturbed_input = input_ + perturbation
       
            # Clamping the audio in original audio range (-1,1)
            perturbed_input = torch.clamp(perturbed_input, -1, 1)

            # Returning perturbed audio
            return perturbed_input.detach().numpy()

        else: # Condition for checking if the user wants 'untargeted' attack to be performed

            # Using the model's transcription as target in untargeted attack
            untarget = list(self.INFER(input_.to(self.device)))
            
            # Encode the target transcription
            encoded_transcription = self._encode_transcription(untarget)

            # Convert the target transcription to a PyTorch tensor
            target = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

            # Computing CTC Loss
            output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
            output = output.transpose(0, 1)
            target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
            loss = F.ctc_loss(output, target, output_length, target_length, blank=0, reduction='mean')

            # Computing gradient of our input w.r.t loss
            loss.backward()

            # If untargeted then we will maximize our loss
            sign_grad = input_.grad.sign() 

            # Calculating 'sign' of the FGSM attack and multiplying it with our small epsilon step
            perturbation = epsilon * sign_grad
            
            # Adding perturbation in the original input to make adversarial example 
            perturbed_input = input_ + perturbation

            # Clamping the audio in original audio range (-1,1)
            perturbed_input = torch.clamp(perturbed_input, -1, 1)

            # Returning perturbed audio
            return perturbed_input.detach().numpy()

    def BIM_ATTACK(self, input__: torch.Tensor, target: List[str] = None,
          epsilon: float = 0.2, alpha: float = 0.1, 
          num_iter: int = 10, nested: bool = True,targeted: bool = False, early_stop: bool = False) -> np.ndarray:

        '''
        Basic Itertive Moment attack is implemented which is simple Fast Gradient 
        Sign Attack but in loop. 
        For more information, see the paper: https://arxiv.org/abs/1607.02533
        
        INPUT ARGUMENTS:
        
        input__       : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                        
        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.
                        
        epsilon       : Maximum allowable noise for our audio.
                        Type: float
                        
        alpha         : Step size for noise to be added in each iteration
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

        early_stop    : If user wants to stop the attack early if the attack reaches the target transcription before the total number of iterations.
                        Type: bool
                        
        RETURNS:
        
        np.ndarray : Perturbed audio
        '''
        
        # Cloning the original given audio
        input_ = input__.clone()
        
        # Making our input differentiable
        input_.requires_grad = True

        # Storing input in variable to add in noise later
        original_input = input_.clone()

        # Checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True
        
        if targeted: # Condition for checking if the user wants 'targeted' attack to be performed

            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)
            
            for i in tqdm(range(num_iter), colour="red", leave = leave):
   
                # Forward pass
                output = self.model(input_.to(self.device)).logits

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Convert the target transcription to a PyTorch tensor
                target_ = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

                # Computing the CTC Loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target_, output_length, target_length, blank=0, reduction='mean')

                # Computing gradients of our input w.r.t loss
                loss.backward()

                # If targeted then we will minimize our loss
                sign_grad = -input_.grad.data 
                
                # Adding perturbation in our input
                perturbed_input = original_input + (alpha * sign_grad.detach().sign())

                # Clamping the perturbation in range (-eps, eps)
                perturbation = torch.clamp(perturbed_input - original_input, -epsilon, epsilon)
                
                # Clamping the overall perturbated audio in the original audio range (-1, 1)
                input_.data = torch.clamp(input_ + perturbation, -1, 1)
                
                if early_stop: # if user have enabled early stopping then do the following tasks or else run all iterations

                    # Storing model's current inference and target transcription in new variables for computing WER
                    string1 = list(filter(lambda x: x!= '',self.INFER(input_).split("|")))
                    string2 = list(reduce(lambda x,y: x+y, target).split("|"))

                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach out target transcription
                    # before the completion of all iterations because further iterations will further
                    # increase noise in the original audio leading to bad/low SNR
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            return input_.detach().numpy()
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            return input_.detach().numpy()
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            return input_.detach().numpy()
                
                # Making gradients of input zero
                input_.grad.zero_()
                
            # Returning perturbed audio after the loop ends
            return input_.detach().numpy()

        else: # Condition for checking if the user wants 'untargeted' attack to be performed
            
            # Using the model's transcription as target in untargeted attack
            target = self.INFER(input_.to(self.device))
            
            for i in tqdm(range(num_iter), colour="red", leave = leave):
                
                # Using the model's transcription as target in untargeted attack
                untarget = self.INFER(input_.to(self.device))
                
                # Encode the target transcription
                encoded_transcription = self._encode_transcription(untarget)
                
                # Forward pass
                output = self.model(input_.to(self.device)).logits

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Convert the target transcription to a PyTorch tensor
                target_ = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

                # Computing the CTC Loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target_, output_length, target_length, blank=0, reduction='mean')

                # Computing gradients of our input w.r.t loss
                loss.backward()

                # If untargeted then we will maximize our loss
                sign_grad = input_.grad.data 

                # Adding perturbation in our input
                perturbed_input = original_input + (alpha * sign_grad.detach().sign())

                # Clamping the perturbation in range (-eps, eps)
                perturbation = torch.clamp(perturbed_input - original_input, -epsilon, epsilon)
                
                # Clamping the overall perturbated audio in the original audio range (-1, 1)
                input_.data = torch.clamp(input_ + perturbation, -1, 1)

                if early_stop: # if user have enabled early stopping then do the following tasks or else run all iterations
                
                    # Storing model's current inference and target transcription in new variables for computing WER
                    string1 = list(self.INFER(input_).split("|"))
                    string2 = list(target.split("|"))
                    
                    # Removing empty spaces (if any) that cause error when computing WER
                    string1 = list(filter(lambda x: x!= '', string1))
                    string2 = list(filter(lambda x: x!= '', string2))

                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach out target transcription
                    # before the completion of all iteration because further iteration will further
                    # increase noise in the original audio leading to bad/low SNR
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            return input_.detach().numpy()
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            return input_.detach().numpy()
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            return input_.detach().numpy()
                
                # Making gradients of input zero
                input_.grad.zero_()
            
            # Returning perturbed audio after the loop ends
            return input_.detach().numpy()

    def PGD_ATTACK(self, input__: torch.Tensor, target: List[str] = None,
                 epsilon: float = 0.3, alpha: float = 0.01, num_iter: int = 40,
                 nested: bool = True,targeted: bool = False, early_stop: bool = False) -> np.ndarray:

        '''
        Projected Gradient Descent attack is implemented which in simple terms is more 
        advanced version of BIM. In this attack we project our perturbation back to 
        some Lp norm and find perturbations in that particular region. 
        For more information, see the paper: https://arxiv.org/abs/1706.06083
        
        INPUT ARGUMENTS:
        
        input__       : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                        
        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]. 
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also.
                        
        epsilon       : Noise controlling parameter.
                        Type: float
                        
        alpha         : Step size for noise to be added in each iteration
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

        early_stop    : If user wants to stop the attack early if the attack reaches the target transcription before the total number of iterations.
                        Type: bool
                        
        RETURNS:
        
        np.ndarray : Perturbed audio
        ''' 
        
        # Cloning the original audio 
        input_ = input__.clone().to(self.device) 
        
        # Making a zero differentiable tensor of same shape as input
        delta = torch.zeros_like(input_, requires_grad=True).to(self.device)
        
        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True

        if targeted: # Condition for checking if the user wants 'targeted' attack to be performed

            # Assert that in targeted attack we have target present before we proceed
            assert target != None, "Target should not be 'None' in targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)
            
            # Convert the target transcription to a PyTorch tensor
            target_ = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()
            
            for i in tqdm(range(num_iter), colour = 'red', leave = leave):

                # Forward pass
                output, _ = self.model(input_ + delta)

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Computing CTC loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target_, output_length, target_length, blank=0, reduction='mean')
                
                # Computing gradients of our input w.r.t loss
                loss.backward()
                
                # Update delta with gradient sign
                sign = -1 # Negative sign because of targeted attack
                delta.data = (delta + alpha * sign * delta.grad.detach().sign())
                
                # Perform projection of delta onto Lp ball
                delta.data = delta.data.clamp(-epsilon, epsilon)

                if early_stop: # if user have enabled early stopping then do the following tasks or else run all iterations
                
                    # Storing model's current inference and target transcription in new variables for computing WER
                    string1 = list(filter(lambda x: x!= '',self.INFER(input_ + delta).split("|")))
                    string2 = list(reduce(lambda x,y: x+y, target).split("|"))

                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach out target transcription
                    # before the completion of all iteration because further iteration will further
                    # increase noise in the original audio leading to bad/low SNR
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                
                # Zeroing the gradients so that they don't accumulate
                delta.grad.zero_()
                
            # Returning perturbed audio after the loop ends
            adv_example = input_ + delta
            return adv_example.detach().cpu().numpy()

        else: # Condition for checking if the user wants 'untargeted' attack to be performed
            
            # We will use the original input's transcription as our target to move away from
            target = self.INFER(input_) 

            for i in tqdm(range(num_iter), colour = 'red', leave = leave):
                
                # We will use the original input's transcription as our target to move away from
                untarget = self.INFER(input_) 

                # Encode the target transcription
                encoded_transcription = self._encode_transcription(untarget)

                # Convert the target transcription to a PyTorch tensor
                target_ = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

                # Forward pass
                output, _ = self.model(input_ + delta)

                # Softmax Activation for computing logits
                output = F.log_softmax(output, dim=-1)

                # Computing CTC loss
                output_length = torch.tensor([output.shape[1]], dtype=torch.long).to(self.device)
                output = output.transpose(0, 1)
                target_length = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(self.device)
                loss = F.ctc_loss(output, target_, output_length, target_length, blank=0, reduction='mean')
                
                # Computing gradients of our input w.r.t loss
                loss.backward()
                
                # Update delta with gradient sign
                sign = 1 # Positive sign because of untargeted attack
                delta.data = (delta + alpha * sign * delta.grad.detach().sign())
                
                # Perform projection of delta onto Lp ball
                delta.data = delta.data.clamp(-epsilon, epsilon)

                if early_stop: # if user have enabled early stopping then do the following tasks or else run all iterations
                
                    # Storing model's current inference and target transcription in new variables for computing WER
                    string1 = list(filter(lambda x: x!= '',self.INFER(input_ + delta).split("|")))
                    string2 = list(reduce(lambda x,y: x+y, target).split("|"))

                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach out target transcription
                    # before the completion of all iteration because further iteration will further
                    # increase noise in the original audio leading to bad/low SNR
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because untargeted Attack is performed successfully !")
                            adv_example = input_ + delta
                            return adv_example.detach().cpu().numpy()
                
                # Zeroing the gradients so that they don't accumulate
                delta.grad.zero_()
                
            # Returning perturbed audio after the loop ends
            adv_example = input_ + delta
            return adv_example.detach().cpu().numpy()

    def CW_ATTACK(self, input__: torch.Tensor, target: List[str] = None,
           epsilon: float = 0.3, c: float = 1e-4, learning_rate: float = 0.01,
           num_iter: int = 1000, decrease_factor_eps: float = 1,
           num_iter_decrease_eps: int = 10, optimizer: str = None, 
           nested: bool = True, early_stop: bool = True, search_eps: bool = False,
           targeted: bool = False, internal_call = False) -> np.ndarray:

        '''
        Implements the Carlini and Wagner attack, the strongest white box 
        adversarial attack. This attack uses an optimization strategy to find the 
        adversarial transcription while keeping the perturbation as low as possible. 
        For more information, see the paper: https://arxiv.org/pdf/1801.01944.pdf
        
        INPUT ARGUMENTS:
        
        input__       : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
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
                        
        decrease_factor_eps   : Factor to decrease epsilon during search
                                Type: float
                                
        num_iter_decrease_eps : Number of iterations after which to decrease epsilon
                                Type: int
                                
        optimizer     : Name of the optimizer to use for the attack. 
                        Type: str
                        
        nested        : if this attack in being run in a for loop with tqdm 
                        Type: bool
                        
        early_stop    : if the user wants to end the attack as soon as the attack
                        gets the target transcription.
                        Type: bool
                        
        search_eps    : if the user wants the attack to search for the epsilon value
                        on its own provided the initial epsilon value of large.
                        Type: bool
        
        targeted      : If the attack should be targeted towards your desired 
                        transcription or not.
                        Type: bool
                        CAUTION:
                        Please make to pass your targetted 
                        transcription also in this case).

        internal_call : If the CW is being called internally by another attack.
                        Type: bool
                        
        RETURNS:
        
        np.ndarray : Perturbed audio
        '''
        
        # checking if user accidentally passed wrong arugments or not
        if early_stop == True and search_eps == True:
            raise Exception("Early stop and Epsilon search arguments, both cannot be true at the same time.")
        
        if epsilon <= 0:
            raise Exception("Value of epsilon should be greater than 0")
        
        # Convert the input audio to a PyTorch tensor
        input_audio = input__.clone().to(self.device).float()

        # Making audio differentiable
        input_audio.requires_grad_()

        # Cloning the original audio 
        input_audio_orig = input_audio.clone().to(self.device)
        
        # Define the optimizer
        if optimizer == "Adam":

            optimizer = torch.optim.Adam([input_audio], lr=learning_rate)

        else:

            optimizer = torch.optim.SGD([input_audio], lr=learning_rate)

        # Setting our inital parameters
        successful_attack = False 
        num_successful_attacks = 0

        # Checking if the user wants to run this code in for loop or not
        if nested:
            leave = False
            descrip = None
            if internal_call:
              descrip = "*"*5+"Attack Stage 1"+"*"*5

        else:
            leave = True
            descrip = None

        if targeted:

            # Making sure target is given in targeted attack
            assert target is not None, "Target should not be 'None' in a targeted attack. Please pass a target transcription."

            # Encode the target transcription
            encoded_transcription = self._encode_transcription(target)
            
            # Convert the target transcription to a PyTorch tensor
            target_tensor = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()
            
            for i in tqdm(range(num_iter), colour="red", leave = leave, desc = descrip):

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
                loss_classifier = F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean')

                # Regularization term to minimize the perturbation
                loss_norm = torch.norm(input_audio - input_audio_orig)

                # Combine the losses and compute gradients
                loss = (c * loss_norm) + ( loss_classifier)

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

                # Storing model's current inference and target transcription in new variables for computing WER
                string1 = list(filter(lambda x: x!= '',self.INFER(input_audio).split("|")))
                string2 = list(reduce(lambda x,y: x+y, target).split("|"))
                
                if early_stop:
                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach our target transcription
                    # before the completion of all iteration
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                        
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                        
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                
                elif search_eps:
                    # Computing WER while also making sure length of both strings is same
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 0:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                            
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                            
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 0:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                
            adv_example = input_audio
            return adv_example.detach().cpu().numpy()
                
        
        else: # If untargeted

            # Then we will use the model's transcription as our target
            target = self.INFER(input_audio.to(self.device)) 
            
            for i in tqdm(range(num_iter), colour="red", leave = leave):
                
                # We will use the model's transcription as our target
                untarget = self.INFER(input_audio.to(self.device)) 

                # Encode the target transcription
                encoded_transcription = self._encode_transcription(untarget)

                # Convert the target transcription to a PyTorch tensor
                target_tensor = torch.from_numpy(np.array(encoded_transcription)).to(self.device).long()

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
                loss_classifier = -F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean')
                
                # Regularization term to minimize the perturbation
                loss_norm = torch.norm(input_audio - input_audio_orig)

                # Combine the losses and compute gradients
                loss = (c * loss_norm) + ( loss_classifier)

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

                # Storing model's current inference and target transcription in new variables for computing WER
                string1 = list(filter(lambda x: x!= '',self.INFER(input_audio).split("|")))
                string2 = list(reduce(lambda x,y: x+y, target).split("|"))

                if early_stop:
                    # Computing WER while also making sure length of both strings is same
                    # This will also early stop the attack if we reach our target transcription
                    # before the completion of all iteration
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                        
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                        
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            print("Breaking for loop because targeted Attack is performed successfully !")
                            adv_example = input_audio
                            return adv_example.detach().cpu().numpy()
                
                elif search_eps:
                    # Computing WER while also making sure length of both strings is same
                    if len(string1) == len(string2):
                        if self._wer(string1, string2)[0] == 1:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                            
                    elif len(string1) > len(string2):
                        diff = len(string1) - len(string2)
                        for i in range(diff):
                            string2.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                            
                    else:
                        diff = len(string2) - len(string1)
                        for i in range(diff):
                            string1.append("<eps>")
                        if self._wer(string1, string2)[0] == 1:
                            num_successful_attacks += 1
                            if num_successful_attacks >= num_iter_decrease_eps:
                                successful_attack = True
                                epsilon *= decrease_factor_eps
                                num_successful_attacks = 0
                            else:
                                successful_attack = False
                
            adv_example = input_audio
            return adv_example.detach().cpu().numpy()

    def IMPERCEPTIBLE_ATTACK(self, input__: torch.Tensor, target: List[str] = None,
                             epsilon: float = 0.3, c: float = 1e-4, learning_rate1: float = 0.01, 
                             learning_rate2: float = 0.01, num_iter1: int = 10000, num_iter2: int = 2000, 
                             decrease_factor_eps: float = 1.0, num_iter_decrease_eps: int = 10, 
                             optimizer1: str = None, optimizer2: str = None, nested: bool = True , 
                             early_stop_cw: bool = True, search_eps_cw: bool = False, alpha: float = 0.5) -> np.ndarray:
        
        '''
        Implements the Imperceptible ASR attack, which leverages the strongest white box 
        adversarial attack which is CW attack while also masking sure the added perturbation
        is imperceptible to humans using Psychoacoustic Scale. This attack is performed in two 
        stages. In first stage we perform simple CW attack and in 2nd stage we make sure our 
        added perturbations are imperceptible. 
        For more information, see the paper: https://arxiv.org/abs/1903.10346
        
        INPUT ARGUMENTS:
        
        input__       : Input audio. Ex: Tensor[0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                        
        target        : Target transcription (needed if the you want targeted 
                        attack) Ex: ["my name is mango."]
                        Type: List[str]
                        CAUTION:
                        Please make sure these characters are also present in the 
                        dictionary of the model also
                        
        epsilon       : Noise controlling parameter
                        Type: float
                        
        c             : Regularization term controlling factor
                        Type: float
                        
        learning_rate1: learning_rate of optimizer for stage 1
                        Type: float
        
        learning_rate2: learning_rate of optimizer for stage 2
                        Type: float
                        
        num_iter1     : Number of iteration of attack stage 1
                        Type: int
        
        num_iter2     : Number of iteration of attack stage 2
                        Type: int
                        
        decrease_factor_eps   : Factor to decrease epsilon by during optimization 
                                Type: float
                                
        num_iter_decrease_eps : Number of iterations after which to decrease epsilon
                                Type: int
                                
        optimizer1     : Name of the optimizer to use for the attack stage 1
                         Type: str
                         
        optimizer2     : Name of the optimizer to use for the attack stage 2
                         Type: str
                         
        nested         : if this attack in being run in a for loop with tqdm 
                         Type: bool
        
        early_stop_cw  : if the user wants 1st stage attack to early stop or not
                         Type: bool
        
        search_eps_cw  : if the user wants 1st stage attack to search for epsilon 
                         value provided a large intial epsilon value is provided
                         Type: bool
                         
        alpha          : regularization term for second stage loss
                         Type: float
                         
        RETURNS:
        
        np.ndarray     : Perturbed audio
        '''
        
        input_ = input__.clone()
        
        # This attack will be targeted for now, therefore...
        assert (target is not None), "Please pass a specific target transcription for performing this attack"

        # checking if the user is running this code in for loop or not
        if nested:
            leave = False

        else:
            leave = True
            
        #stage 1 of Imperceptible ASR attack
        stageOneAud =  self.CW_ATTACK(input_ , target = target, epsilon = epsilon, c = c, learning_rate = learning_rate1,
                           num_iter = num_iter1, decrease_factor_eps = decrease_factor_eps, 
                           num_iter_decrease_eps = num_iter_decrease_eps, optimizer = optimizer1, 
                           nested = True, early_stop = early_stop_cw, search_eps = search_eps_cw, targeted = True, internal_call = True)

        
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
        theta = torch.tensor(theta.transpose(1, 0)).to(self.device)
          
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
        
        losss = []
        examplee = []
        
        #stage 2 of Imperceptible ASR attack
        for i in tqdm(range(num_iter2), colour = 'red', leave = leave, desc="*"*5+"Attack Stage 2"+"*"*5):
                      
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
            loss_classifier = F.ctc_loss(output, target_tensor, output_length, target_length, blank=0, reduction='mean')
            
            # Regularization term to minimize the perturbation
            loss_regularizer = torch.norm(input_audio - input_audio_orig)

            # Combine the losses and compute gradients
            loss1 = ( torch.tensor(c) * loss_classifier) + (loss_regularizer)
            
            # Calculating perturbation by subtracting the optimized audio from cloned one
            perturbation = input_audio - input_audio_orig
            
            # Calculating PSD matrix of perturbation which is added in the clean audio
            psd_transform_delta = self._psd_transform(
                delta=perturbation, original_max_psd=original_max_psd
            )
            
            # Calculating the loss between perturbation's PSD and original audio's PSD
            loss2 = torch.mean(relu(psd_transform_delta - theta))
            
            # Summing both CW loss and PSD loss
            loss = loss1.type(torch.float64) + (torch.tensor(alpha).to(self.device) * loss2)
            
            # Taking mean of both losses
            loss = torch.mean(loss)
            
            # Computing gradients of our input w.r.t loss
            loss.backward()

            # Update the input audio with gradients
            optimizer.step()
            
            # Storing model's current inference and target transcription in new variables for computing WER
            string1 = list(filter(lambda x: x!= '',self.INFER(input_audio).split("|")))
            string2 = list(reduce(lambda x,y: x+y, target).split("|"))
            
            if len(losss) <= 300: #buffer logic
                
                losss.append(loss)
                examplee.append(input_audio.detach().cpu().numpy())
            
            else:
                
                del losss[0]
                del examplee[0]
                losss.append(loss)
                examplee.append(input_audio.detach().cpu().numpy())
            
            if i % 20 == 0: # if every 20 iterations the transcription matches then alpha value will be increased
        
                if len(string1) == len(string2):
                    if self._wer(string1, string2)[0] == 0:
                        alpha = alpha * 1.2

                elif len(string1) > len(string2):
                    diff = len(string1) - len(string2)
                    for i in range(diff):
                        string2.append("<eps>")
                    if self._wer(string1, string2)[0] == 0:
                        alpha = alpha * 1.2

                else:
                    diff = len(string2) - len(string1)
                    for i in range(diff):
                        string1.append("<eps>")
                    if self._wer(string1, string2)[0] == 0:
                        alpha = alpha * 1.2
            
            if i % 50 == 0: # if every 50 iterations the transcription does not match then alpha value will be decreased
        
                if len(string1) == len(string2):
                    if self._wer(string1, string2)[0] != 0:
                        alpha = alpha * 0.8
  
                elif len(string1) > len(string2):
                    diff = len(string1) - len(string2)
                    for i in range(diff):
                        string2.append("<eps>")
                    if self._wer(string1, string2)[0] != 0:
                        alpha = alpha * 0.8

                else:
                    diff = len(string2) - len(string1)
                    for i in range(diff):
                        string1.append("<eps>")
                    if self._wer(string1, string2)[0] != 0:
                        alpha = alpha * 0.8
        
        #return the example with the lowest loss among the stored examples
        minimumLoss = min(losss) 
        indexLoss = losss.index(minimumLoss)
        adv_example = examplee[indexLoss]
        return adv_example 
               
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
        delta_stft = torch.view_as_real(torch.stft(
          delta,
          n_fft=2048,
          hop_length=512,
          win_length=2048,
          center=False,
          window=window_fn(2048).to(self.device),
          return_complex=True,
        )).to(self.device)

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

    def _wer(self, reference, prediction) -> Tuple[int, Tuple[int, int, int]]:

        '''
        This method only for use internally that's why I have used underscore before the method.
        This method calculates word error rate of of a single example provided it is given equal length transcription.
        If transcriptions are not equal, make them equal by appending <eps> in which ever transcription who's length is smaller than the other.

        INPUT ARGUMENTS:

        reference     : Ground Truth. Ex ['My', 'name', 'is', 'Bond']
                        Type: List

        prediction    : Model's output transcription. Ex ['My', 'name', 'is', 'Band']
                        Type: List

        RETURNS:

        Tuple[int, Tuple[int, int, int]] : single transcription's WER along with another tuple containing information of (Substitution, Insertion, Deletion)
        '''

        correct = 0
        substitution = 0
        insertion = 0
        deletion = 0
        for i in range(len(reference)):
            if prediction[i] == reference[i]:
                correct +=1
            elif prediction[i] != reference[i] and prediction[i] != '<eps>' and reference[i] != '<eps>':
                substitution+=1
            elif prediction[i] == '<eps>':
                deletion+=1
            elif prediction[i] != reference[i] and reference[i] == '<eps>':
                insertion+=1
        wer = (substitution + insertion + deletion) / (correct + substitution + deletion + insertion)
        return wer, (substitution, insertion, deletion)
    
    def INFER(self, input_: torch.Tensor) -> str:
        
        '''
        Method for performing inference by the model.
        
        INPUT ARGUMENTS:
        
        input_        : Input audio of shape Ex: [0.1,0.3,...] or (samples,)
                        Type: torch.Tensor
                         
        RETURNS:
        
        str           : Model's transcription from the given audio.
        '''

        # Inference from the model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_.to(device)).logits
            logits = F.log_softmax(logits, dim=-1)

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode
        return transcription[0]

    def wer_compute(self, ground_truth: List[str], audios: List[np.ndarray], targeted:bool = False)-> Tuple[int, List[Tuple[int, int ,int]]]:

        '''
        Computes WER of a single audio or batch of audios

        INPUT ARGUMENTS:

        ground_truth : Original transcription or reference transcription.
                        Type: List[str]

        audios       : Audios who's transcription from model will be used.
                        Type: List[np.ndarray]

        targeted     : if the WER should be computed according to the way for
                        targeted attack or not.

        RETURNS:

        Tuple[int, List[Tuple[int, int ,int]]] : average WER of given audio/audios and List of Tuple which contains information of (Substitution, Insertion, and Deletion) of every audio separately passed in a batch
        '''

        wer_count = 0
        sib_saver = []
        if targeted == True:
            for i in range(len(audios)):
                prediction = self.INFER(torch.from_numpy(audios[i]))
                reference  = ground_truth[i].split(" ")
                prediction = list(filter(lambda x: x!='', prediction.split("|")))
                if len(prediction) == len(reference):
                    word_error_rate = max(1 - self._wer(reference, prediction)[0], 0)
                    sib_saver.append(self._wer(reference, prediction)[1])
                elif len(prediction) > len(reference):
                    diff = len(prediction) - len(reference)
                    for _ in range(diff):
                        reference.append("<eps>")
                    word_error_rate = max(1 - self._wer(reference, prediction)[0], 0)
                    sib_saver.append(self._wer(reference, prediction)[1])
                else:
                    diff = len(reference) - len(prediction)
                    for _ in range(diff):
                        prediction.append("<eps>")
                    word_error_rate = max(1 - self._wer(reference, prediction)[0], 0)
                    sib_saver.append(self._wer(reference, prediction)[1])
                wer_count += word_error_rate
                if i == len(audios) - 1:
                    return wer_count/len(audios), sib_saver
        else:
            for i in range(len(audios)):
                prediction = self.INFER(torch.from_numpy(audios[i]))
                reference  = ground_truth[i].split(" ")
                prediction = list(filter(lambda x: x!='', prediction.split("|")))
                if len(prediction) == len(reference):
                    word_error_rate = min(self._wer(reference, prediction)[0], 1)
                    sib_saver.append(self._wer(reference, prediction)[1])
                elif len(prediction) > len(reference):
                    diff = len(prediction) - len(reference)
                    for _ in range(diff):
                        reference.append("<eps>")
                    word_error_rate = min(self._wer(reference, prediction)[0], 1)
                    sib_saver.append(self._wer(reference, prediction)[1])
                else:
                    diff = len(reference) - len(prediction)
                    for _ in range(diff):
                        prediction.append("<eps>")
                    word_error_rate = min(self._wer(reference, prediction)[0], 1)
                    sib_saver.append(self._wer(reference, prediction)[1])
                wer_count += word_error_rate
                if i == len(audios) - 1:
                    return wer_count/len(audios), sib_saver