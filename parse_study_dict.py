import SimpleITK as sitk
import json

class ParserStudyDict:
    def __init__(self,studyDict):
        self.dict = studyDict
  
        self.id                         = None
        self.fixed_filename             = None
        self.fixed_segmentation_filename= None
        self.moving_type                = None
        self.moving_filename            = None
        self.moving_dict                = None
        self.fIC                        = None
        self.DWI                        = None
        self.DWI_map                    = None
        self.cancer                     = None
        self.landmarks                  = None
        
        self.SetFromDict()
        
    def SetFromDict(self):  
        try:
            self.fixed_filename                 = self.dict['fixed']
        except :
            pass
            
        try:
            self.fixed_segmentation_filename    = self.dict['fixed-segmentation']
        except :
            pass


        try:
            self.moving_type                    = self.dict['moving-type']

        except :
            pass

        try:
            self.moving_filename                = self.dict['moving']
        except :
            pass
            
        try:
            self.id                             = self.dict['id']
        except :
            pass
        
        try:
            DWI                                 = self.dict['DWI']
            if DWI == "True":
                self.DWI = True                           
        except :
            pass
        
        try:
            self.DWI_map                        = self.dict['DWI-map']                   
        except :
            pass
            
        try:
            self.fIC                            = self.dict['fIC']                   
        except :
            pass
            
        try:
            self.cancer                         = self.dict['cancer']                   
        except :
            pass
        
        try:
            self.landmarks                      = self.dict['landmarks']                   
        except :
            pass
            
            
    def ReadImage(self, fn):
        im = None
        if fn:
            try:
                im = sitk.ReadImage( fn )
            except :
                pass
                print("Fixed image cound not be read from", fn)
                im = None
        else:
            print("Fixed filename is not available and fixed image cound't be read")
            im = None
            
        return im
            
    def ReadMovingImage(self):   
        
        if self.moving_type and self.moving_type.lower()=="stack":
            #print(self.moving_filename)
            with open(self.moving_filename) as f:
                self.moving_dict = json.load(f)
        else:
            self.moving_dict = None
        
        return self.moving_dict