from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from torch.utils.tensorboard import SummaryWriter



def FindValueOfClass(line, stringToFind):
    idxStringToFind = line.find(stringToFind)
    idxNumber = idxStringToFind + len(stringToFind) - 1
    valueString = line[idxNumber:idxNumber + 9]
    valueFloat = float(valueString) 
    return valueFloat  


def main():
    print("=======================\nParser started...\n=======================\n")

    parser = argparse.ArgumentParser(description='Log to Tensorboard Parser')
    parser.add_argument('--log_path', type=str, default='outputs/', metavar='N',
                            help='Filepath of the log file to parse')
    parser.add_argument('--output_path', type=str, default='parser_output/', metavar='N',
                            help='Output path where the tensorboard file is saved')
    args = parser.parse_args()
    print("Log file path: " + args.log_path)
    infile = args.log_path
    output_path = args.output_path

    with open(infile) as f:
        f = f.readlines()

    trainLossValues = []
    trainAccValues = []
    trainAvgValues = []
    trainIouValues = []

    testLossValues = []
    testAccValues = []
    testAvgValues = []
    testIouValues = []

    for line in f:
        first_word = line.split()[0]
        if (first_word == "Train"):
            stringToFind = "loss: "
            trainLossValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "train acc: "
            trainAccValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "train avg acc: "
            trainAvgValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "train iou: "
            trainIouValues.append(FindValueOfClass(line, stringToFind))
        elif (first_word =="Test"):
            stringToFind = "loss: "
            testLossValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "test acc: "
            testAccValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "test avg acc: "
            testAvgValues.append(FindValueOfClass(line, stringToFind))
            stringToFind = "test iou: "
            testIouValues.append(FindValueOfClass(line, stringToFind))
        else:
            continue

    writer = SummaryWriter(output_path)

    # ==== TRAIN ====
    for i, value in enumerate(trainLossValues):
        writer.add_scalar("Loss/Train", value, i)
    for i, value in enumerate(trainAccValues):
        writer.add_scalar("Accuracy/Train", value, i)
    for i, value in enumerate(trainAvgValues):
        writer.add_scalar("Average Accuracy/Train", value, i)
    for i, value in enumerate(trainIouValues):
        writer.add_scalar("IOU/Train", value, i)
    
    # ==== TEST ====
    for i, value in enumerate(testLossValues):
        writer.add_scalar("Loss/Test", value, i)
    for i, value in enumerate(testAccValues):
        writer.add_scalar("Accuracy/Test", value, i)
    for i, value in enumerate(testAvgValues):
        writer.add_scalar("Average Accuracy/Test", value, i)
    for i, value in enumerate(testIouValues):
        writer.add_scalar("IOU/Test", value, i)

    print("Parsing successful!\nOutput at: " + output_path)


if __name__ == "__main__":
    main()
