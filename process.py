import torch
import os
import torchio as tio
from inference import inference

class Mv1():
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.ckpt_path = '/opt/algorithm/Mv1_comp_PTCT.pt'
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        self.gpu_available = False
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))
            self.gpu_available = True

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]
        
        ct_path = os.path.join(self.input_path, 'images/ct/', ct_mha)
        pt_path = os.path.join(self.input_path, 'images/pet/', pet_mha)
        
        subject = tio.Subject(
            CT=tio.ScalarImage( ct_path ),
            PT=tio.ScalarImage( pt_path ),
        )

        return subject, uuid

    def write_outputs(self, subject, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        
        out_path = os.path.join(self.output_path, uuid + ".mha")
        subject.prediction.save(out_path)
        
        print('Output written to: ' + out_path)
    
    def predict(self, inputs):
        """
        Your algorithm goes here
        """        
        pass
        #return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        device='cpu'
        if self.gpu_available:
            device='cuda:0'
        
        config = {'patch_size': 64,
                  'stride': 32,
                  'bs_test': 16,
                  'n_classes': 2,
                  'thres_prob':0.5,
                  'device': device
                  }
        
        print('Start processing')
        subject,uuid = self.load_inputs()
        print('Start prediction')
        segm = inference(self.ckpt_path, subject,config)
        subject.add_image(tio.LabelMap(tensor=segm,affine=subject.PT.affine),'prediction')
        print('Start output writing')
        self.write_outputs(subject,uuid)
        
        
if __name__ == "__main__":
    Mv1().process()
