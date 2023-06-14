import os
import cv2 as cv
import torch
import numpy as np
from torchvision import transforms
from surface_detect_model import SurfaceDefectResNet

defect_labels = ["In", "Sc", "Cr", "PS", "RS", "Pa"]


def defect_demo():
    cnn_model = SurfaceDefectResNet()
    cnn_model.load_state_dict(torch.load("D:/models/surface_defect_model.pt"))
    cnn_model.eval()
    cnn_model.cuda()
    print(cnn_model)
    root_dir = "D:/datasets/enu_surface_defect/test"
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        transforms.Resize((200, 200))])
    filenames = os.listdir(root_dir)
    for file in filenames:
        image = cv.imread(os.path.join(root_dir, file))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        x_input = img_transform(image).view(1, 3, 200, 200)
        probs = cnn_model(x_input.cuda())
        predic_ = probs.view(6).cpu().detach().numpy()
        idx = np.argmax(predic_)
        defect_txt = defect_labels[idx]
        print(defect_txt, file)
        cv.putText(image, defect_txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (30, 120, 200), 2, 8)
        cv.imshow("defect_detection", image)
        cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    defect_demo()
