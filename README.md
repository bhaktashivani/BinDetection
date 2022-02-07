Name: Shivani Bhakta 
I have created a separate file name train_pixel_classifier.py in pixel_classifier folder. 
I import it in pixer.classifier.py and it doen't need to run on its on. 

I have created a separate file name generate_bin_data.py in bin_detection folder. 
I import used it to label and get my mask, doesn't need to run on its own either.

Saving my weights in .csv files, that are being called in both bin_detector.py and pixel_classifier.py. So both of them need to be there when running those files. (they both) 
are in their respective folder.  


