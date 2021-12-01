# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 20:08:16 2021

@author: Sagar Mordiya
"""
# pip install pandas
import pandas as pd
# pip install opencv-python
import cv2
# pip install pyabf
import pyabf


def ftnReadImage(iFile):
    cv2.imshow('Image',imgFile)
    k = cv2.waitKey(0)
    # wait for ESC key to exit
    if k == 27:
       cv2.destroyAllWindows()

def ftnReadCSV(cFile):
    print(csvFile)

def ftnReadBinary(bFile):
    print(binaryFile)

def ftnReadExcel(eFile):
    print(excelFile)

def ftnReadPython(pFile):
    body = pythonFile.read()
    print(body)
    

def ftnSwitchCase(i, c, b, e, p):
    print("Operation: Read the files:")
    print("1.Image \t 2.CSV \t 3.Binary \t 4.Excel \t 5.Python")
    try:
        intOptn = int(input("Enter the choice (0 for Exit): "))
    except ValueError:
        print("Enter Value between 1 to 5")
    while intOptn != 0:
        
        if intOptn == 0:
            exit
            
        elif intOptn == 1:
            ftnReadImage(i)
    
        elif intOptn == 2:
            ftnReadCSV(c)
    
        elif intOptn == 3:
            ftnReadBinary(b)
    
        elif intOptn == 4:
            ftnReadExcel(e)
    
        elif intOptn == 5:
            ftnReadPython(p)
            
        else:
            print("Invalid option, Please try again!")
        
        intOptn = int(input("Enter the choice (0 for Exit): "))
        
        

    
    
if __name__ == "__main__":
    # Importing All required files
    imgFile = cv2.imread("Nuclei_Msk.jpg")
    csvFile = pd.read_csv("Nuclei.csv")
    binaryFile = pyabf.ABF("16o03002.abf")
    excelFile = pd.read_excel("Financial Sample.xlsx")
    pythonFile = open("template.py","r")
    ftnSwitchCase( imgFile, csvFile, binaryFile, excelFile, pythonFile)
