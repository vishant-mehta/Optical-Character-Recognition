# Optical-Character-Recognition

Despite decades of research, developing optical character recognition (OCR) systems with capabilities comparable to that of a human remains an open challenge. A large scale of documents in the form of the image is needed to be entered into computer databases which takes a lot of memory as compared to editable text and there can be errors while interpretation of data from an image. This project aims to use OCR to convert handwritten or printed documents into editable text. 

Documents are scanned to image format as an input to a doc class net which is a full size image classifier that classifies the input image into four different classes viz. printed, semi-printed, handwritten discrete, and handwritten cursive. The OCR model predicts and then decodes the text in the image and gives the output as an editable text. 

We have applied OCR to printed text images using the Pytesseract. For handwritten text images, the text is predicted using a self-developed convolutional recurrent neural network (CRNN) named CL-9 (7 CNN layers and 2 LSTM layers). 

The accuracy of the doc class net classifier and line class net classifier(line wise classifier) was 88.03 % and 82.1 % respectively. The overall accuracy for printed, handwritten discrete and handwritten cursive obtained is 94.79 %, 75.2 %, and 65.7 % respectively. 

OCR has real-time applications in various fields like medical prescriptions, smart libraries, and tax returns.

Research paper: Optical Character Recognition Using Deep Learning Techniques for Printed and Handwritten Documents

Link: https://ssrn.com/abstract=3664620 (Elsevier SSRN Repository)
