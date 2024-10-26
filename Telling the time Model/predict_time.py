import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
# include whatever other imports you need here

class TimePredictionNetwork(nn.Module):
    
   # Your network definition goes here
   def __init__(self):

    super(TimePredictionNetwork, self).__init__()

    self.convlayers = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels=32, kernel_size = 3, stride =1, padding =1),
        # B x 32 x 64 x 64
        nn.BatchNorm2d(32),
        # B x 32 x 64 x 64
        nn.ReLU(),
        # B x 32 x 64 x 64
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        # B x 32 x 32 x 32
        nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride =1, padding =1),
        # B x 64 x 32 x 32
        nn.BatchNorm2d(64),
        # B x 64 x 32 x 32
        nn.ReLU(),
        # B x 64 x 32 x 32
        nn.MaxPool2d(kernel_size=2,stride =2),
        # B x 64 x 16 x 16
        nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = 3,stride =1 , padding =1),
        # B x 128 x 16 x 16
        nn.BatchNorm2d(128),
        # B x 128 x 16 x 16
        nn.ReLU(),
        # B x 128 x 16 x 16
        nn.MaxPool2d(kernel_size=2,stride =2),
      # B x 128 x 8 x 8
        nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3,stride =1 , padding =1),
        # B x 256 x 8 x 8
        nn.BatchNorm2d(256),
        # B x 256 x 8 x 8
        nn.ReLU(),
        # B x 256 x 8 x 8
        nn.MaxPool2d(kernel_size=2,stride =2),
        # B x 256 x 4 x 4
        nn.Dropout(0.75)


    )

    self.fc1 = nn.Linear(in_features = 4 * 4 * 256, out_features = 1024)

    self.batch1 = nn.BatchNorm1d(1024)

    self.relu =  nn.ReLU()

    self.fc2 =  nn.Linear(in_features = 1024, out_features = 512)

    self.batch2 = nn.BatchNorm1d(512)

    self.dropout2 = nn.Dropout(0.25)

    self.fcHours = nn.Linear(in_features = 512, out_features= 12)

    self.fcMins = nn.Linear(in_features = 512, out_features = 60)

  def forward(self,x):

    x= self.convlayers(x)

    x = x.view(x.size(0), -1)

    x = self.fc1(x)

    x = self.batch1(x)

    x = self.relu(x)

    x = self.fc2(x)

    x = self.batch2(x)

    x = self.relu(x)

    x = self.dropout2(x)

    hours = self.fcHours(x)
    mins = self.fcMins(x)

    return hours, mins

def predict(images):
    
    device = torch.device("cuda" if images.is_cuda else "cpu")

    model = TimePredictionNetwork()
    model = model.to(device)
    model.load_state_dict(torch.load('model_weights_final.pkl', map_location=device))
    model.eval()

   
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    
    images_resized = torch.zeros((images.size(0), 3, 64, 64), device=device)
    for i in range(images.size(0)):
        image = transforms.ToPILImage()(images[i].cpu())
        image = transform(image)
        images_resized[i] = image



    with torch.no_grad():
        hours_output, mins_output = model(images_resized)
        pred_hours = torch.argmax(hours_output, dim=1)
        pred_mins = torch.argmax(mins_output, dim=1)
        predicted_times = torch.stack((pred_hours, pred_mins), dim=1)

    return predicted_times
