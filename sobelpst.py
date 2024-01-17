import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# img2=cv2.imread('/home/psj/datasets/ClothesRGBD/Output_bigroom1/Camera1_0/segments/6/input_Cam000.png')
# img=cv2.imread('/home/psj/datasets/ClothesRGBD/Output_bigroom1/Camera1_0/segments/6/input_Cam000.png')
# img1=cv2.imread('/home/psj/datasets/ClothesRGBD/Output_bigroom1/Camera1_0/segments/6/input_Cam000.png',　 cv2.IMREAD_GRAYSCALE)
from torchvision import transforms
import torch
import numpy as np
# torch.set_printoptions(profile="full")

with open(os.path.join('/home/user/Segmentation_final3/PST900/PST900_RGBT_Dataset/', f'test.txt'), 'r') as f:
    image_labels = f.readlines()
    # print(image_labels)
for i in range(len(image_labels)):
    label_path1 = image_labels[i].strip()
    # print(label_path1)
    # image_path, label_path = image_labels[i].strip().split(' ')
    imgrgb= cv2.imread('/home/user/Segmentation_final3/PST900/PST900_RGBT_Dataset/test/rgb/' + label_path1 + '.png' , 0)
    imgdepth = cv2.imread('/home/user/Segmentation_final3/PST900/PST900_RGBT_Dataset/test/thermal/' + label_path1 + '.png', 0)
    # bound = cv2.imread('/home/user/Segmentation_final3/irseg/bound/' + label_path1 + '.png', 0)
    # edge = cv2.imread('/home/user/Segmentation_final3/irseg/edge/' + label_path1 + '.png', 0)
    # # print(imgrgb)
    # # print(bound.shape)
    # print(edge.shape)

    def tensor_to_PIL(tensor):
        # image = tensor.cpu().clone()
        image = tensor.squeeze(0)
        image = unloader(image)
        # print(image.size())
        return image
    #



    x1 = cv2.Sobel(imgrgb, cv2.CV_16S, 1, 0)
    y1 = cv2.Sobel(imgrgb, cv2.CV_16S, 0, 1)
    x2 = cv2.Sobel(imgdepth, cv2.CV_16S, 1, 0)
    y2 = cv2.Sobel(imgdepth, cv2.CV_16S, 0, 1)

    absX1 = cv2.convertScaleAbs(x1)  # 转回uint8
    absY1 = cv2.convertScaleAbs(y1)
    absX2 = cv2.convertScaleAbs(x2)  # 转回uint8
    absY2 = cv2.convertScaleAbs(y2)

    dst1 = cv2.addWeighted(absX1, 0.5, absY1, 0.5, 0)
    dst2 = cv2.addWeighted(absX2, 0.5, absY2, 0.5, 0)
    # print(dst1)

    # print(dst1)
    # dst1 = torch.from_numpy(dst1)
    # print(dst1)



    # print(dst)
    # print(dst.dtype)
    # print(dst / 255)
    #
    # x3 = cv2.Sobel(dst, cv2.CV_16S, 1, 0)
    # y3 = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    # absX3 = cv2.convertScaleAbs(x3)  # 转回uint8
    # absY3 = cv2.convertScaleAbs(y3)
    # dst3 = cv2.addWeighted(absX3, 0.5, absY3, 0.5, 0)


    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)

    # cv2.imshow("Result1", dst1)
    # cv2.imshow("Result2", dst2)
    # cv2.imshow("Result", dst)
    # cv2.imshow("Result3", dst3)
    # # print(dst)
    # #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    loader = transforms.Compose([
        transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    # a = loader(dst)
    # print(a)

    # print(bound.max())


    dst1 = loader(dst1)
    dst2 = loader(dst2)
    # print(dst1)
    dst = (dst1 + dst2)
    # a = torch.ones(1, 480, 640)
    # b = torch.zeros(1, 480, 640)
    #
    # # c = torch.where(dst > 0.2, dst, b)
    # c = torch.where(dst < 0.2, dst, a)
    # c = torch.where(c > 0.2, c, b)
    # # c = torch.where(c > 0.7, c, b)
    # # print(c)
    # c = c / 255.


    c = tensor_to_PIL(dst)
    # print(c)
    c = np.array(c)
    # c = c / 255.
    # print(c.shape)
    #
    # cv2.imshow("Result3", c)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # c.show()
    # break


    # print(dst.shape)
    # print(dst.max())
    # print(bound)
    # for i in range(640):
    #     for j in range(480):
    #         if dst[i][j] > 0.5:
    #             dst[i][j] = dst[i][j] + 0.3






    # # print(a)
    # #
    #
    # def tensor_to_PIL(tensor):
    #
    #
    #     image = tensor.cpu().clone()
    #
    #     image = image.squeeze(0)
    #
    #
    #     image = unloader(image)
    #
    #
    #
    #     return image
    #
    # image = tensor_to_PIL(a)


    # image1 = tensor_to_PIL(dst1)
    # image2 = tensor_to_PIL(dst2)
    # image = tensor_to_PIL(dst)
    # image1.show()
    # image2.show()
    # image.show()

    # x3 = cv2.Sobel(dst, cv2.CV_16S, 1, 0)
    # y3 = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    # absX3 = cv2.convertScaleAbs(x3)  # 转回uint8
    # absY3 = cv2.convertScaleAbs(y3)
    # dst3 = cv2.addWeighted(absX3, 0.5, absY3, 0.5, 0)
    #
    # cv2.imshow("Result3", dst3)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # print(image.shape)
    # # print(image.shape)
    # #
    # #
    # # def image_loader(image_name):
    # #     image = Image.open(image_name).convert('RGB')
    # #     image = loader(image).unsqueeze(0)
    # #     return image.to(device, torch.float)
    # break

    cv2.imwrite('/home/user/Segmentation_final3/PST900/PST900_RGBT_Dataset/testedge/' + label_path1 + '.png', c)
    # c.save(os.path.join('/home/user/Segmentation_final3/irseg/edge3/' + label_path1 + '.png'))
    # print(c.shape)
    #
    # break
    # cv2.imwrite('ft_gray.png', Grayimg)
    # plt.imshow(img_binary)
    # plt.show()
    # cv2.imwrite('ft.png', img_binary)


