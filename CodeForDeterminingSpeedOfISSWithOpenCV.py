1 """
2 Here comes first the code from the participants, then follows the pupils work
with the code.
3 That is the pupils commenting of the central parts of the code.
4 The pupils have tested an discussed changes and improvements to the code in
collaboration with the teacher.
5 The results of the code is saved in the file named 'results.txt'
6 The focus here has been the correlation between the calculation of the speed
and the percentage of clouds
7 in the images taken from the ISS.
8 So that is why the results file is filled with collumns one with the
calculated speed
9 and the next one with the calculated percantage of clouds in the two images
used for the speed calculation.
10 x : matrix of x positions
11
 """

if __name__ == "__main__":

    from exif import Image

    from datetime import datetime

    from datetime import timedelta

    

    from time import sleep

    import cv2

    import math

    

    from picamera import PiCamera

    

    from pathlib import Path

    

    

    cam=PiCamera()

    cam.resolution = (4056, 3040)

    

    dir_path = Path(__file__).parent.resolve()#the path to where this porgam sits on the ISS astro pi is found and sat up in a varaiable.

    

    start_time = datetime.now()#timing variables are sat up here 

    now_time = datetime.now()

    #duration = datetime.timedelta(minutes=1)#10780)#Here the time limit is sat up just under the max of thre hours (3*3600-20)


    

    

    

    def get_time(image):

        with open(image, 'rb') as image_file:

            img = Image(image_file)

            for data in img.list_all():

#                print(data)

                time_str = img.get("datetime_original")

            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')

        return time

    #print(get_time('photo_0683.jpg'))

    def get_time_difference(image_1, image_2):

        time_1 = get_time(image_1)

        time_2 = get_time(image_2)

        time_difference = time_2 - time_1

        return time_difference.seconds

        #print(time_difference)

    #get_time_difference('photo_0683.jpg', 'photo_0684.jpg')

    

    def convert_to_cv(image_1, image_2):

        image_1_cv = cv2.imread(image_1, 0)

        image_2_cv = cv2.imread(image_2, 0)

        return image_1_cv, image_2_cv

    

    def calculate_features(image_1, image_2, feature_number):

        orb = cv2.ORB_create(nfeatures = feature_number)

        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)

        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)

        return keypoints_1, keypoints_2, descriptors_1, descriptors_2

    def calculate_matches(descriptors_1, descriptors_2):

        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = brute_force.match(descriptors_1, descriptors_2)

        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    

    

    

    

    

    def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):

        match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)

        resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)

        cv2.imshow('matches', resize)

        cv2.waitKey(0)

        cv2.destroyWindow('matches')

    

    def find_matching_coordinates(keypoints_1, keypoints_2, matches):

        coordinates_1 = []

        coordinates_2 = []

        for match in matches:

            image_1_idx = match.queryIdx

            image_2_idx = match.trainIdx

            (x1,y1) = keypoints_1[image_1_idx].pt

            (x2,y2) = keypoints_2[image_2_idx].pt

            coordinates_1.append((x1,y1))

            coordinates_2.append((x2,y2))

        return coordinates_1, coordinates_2

    

    def calculate_mean_distance(coordinates_1, coordinates_2):

        all_distances = 0

        merged_coordinates = list(zip(coordinates_1, coordinates_2))

        for coordinate in merged_coordinates:

            x_difference = coordinate[0][0] - coordinate[1][0]

            y_difference = coordinate[0][1] - coordinate[1][1]

            distance = math.hypot(x_difference, y_difference)

            all_distances = all_distances + distance

        return all_distances / len(merged_coordinates)

        #print(coordinates_1[0])

        #print(coordinates_2[0])

        #print(merged_coordinates[0])

    def calculate_speed_in_kmps(feature_distance, GSD, time_difference):

        distance = feature_distance * GSD / 100000

        speed = distance / time_difference

        return speed

    

    #Here comes the landseadetection imports and funcionsdefinitions:

    

    import numpy as np

    import cv2

    
    def find_border(img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (0,0,0), (180,255,45))

        output = cv2.bitwise_and(img, img, mask=mask)

        return output

    

    def find_sea(img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#lav billed om fra RGB til HSV

        mask = cv2.inRange(hsv, (90,50,50), (125,255,255))#sortere pixel ud fra en tærskel værdi - returnere et sort hvidt billede

        output = cv2.bitwise_and(img, img, mask=mask)#returnere billede der viser pixel der har værdi inden for tærskel område, alle andre pixel er sorte.

        return output

    

    def find_cloud(img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (0,0,120), (180,30,255))

        output = cv2.bitwise_and(img, img, mask=mask)

        return output

    

    def find_ground(img, border, sea, cloud):

        output = cv2.bitwise_xor(img, border)

        output = cv2.bitwise_xor(output, sea)

        output = cv2.bitwise_xor(output, cloud)

        return output

    

    def compute_percentage(masked_img):

        single_band_img= masked_img[:,:,0]>0

        npixel_class = sum(sum(1*single_band_img))

        return npixel_class

    

    def vis_billed(img,name):

        cv2.namedWindow(name,cv2.WINDOW_NORMAL)

        cv2.imshow(name,img)

        cv2.resizeWindow(name, 600,600)

    

    def calculate_percentage(border_output,sea_output,cloud_output,ground_output):

        nborder_pixel = compute_percentage(border_output)

        nsea_pixel = compute_percentage(sea_output)

        ncloud_pixel = compute_percentage(cloud_output)

        nground_pixel = compute_percentage(ground_output)

        total_img_pixel = img[:,:,0].shape[0]*img[:,:,0].shape[1]

        return [nborder_pixel/total_img_pixel, nsea_pixel/(total_img_pixel-nborder_pixel), ncloud_pixel/total_img_pixel, nground_pixel/(total_img_pixel-nborder_pixel)]

    

    def global_classificator(img):

        border_output = find_border(img)

        sea_output = find_sea(img)

        cloud_output = find_cloud(img)

        ground_output = find_ground(img, border_output, sea_output, cloud_output)

    

#        vis_billed(border_output,'border')

 #       vis_billed(sea_output,'sea')

  #      vis_billed(cloud_output,'cloud')

   #     vis_billed(ground_output,'ground')
    #    vis_billed(img,'original image')

        #print('border, sea, cloud, ground')

        #print(calculate_percentage(border_output,sea_output,cloud_output,ground_output))
        return calculate_percentage(border_output,sea_output,cloud_output,ground_output)


    

    
    # Run a loop for 1 minute
    speedlist=[]
    while (now_time < start_time + timedelta(seconds=20)):#600)): 10 minutes 600=10*60

        #print("Hello from the ISS by 2 seconds")

#        sleep(1)

        # Update the current time

        now_time = datetime.now()

    # Out of the loop — stopping

    

    

    #image_1 = 'photo_0683.jpg'

    #image_2 = 'photo_0684.jpg'

        cam.capture("Nycoast.jpg")

        img = cv2.imread('Nycoast.jpg')

        percentage1=global_classificator(img)
    

        #image_1 = cv2.imread('Nycoast.jpg')

    

        cam.capture("Nycoast2.jpg")

        img = cv2.imread('Nycoast2.jpg')

        percentage2=global_classificator(img)
    

        #image_2 = cv2.imread('Nycoast2.jpg')

        image_1 = 'Nycoast.jpg'

        image_2 = 'Nycoast2.jpg'

    

    

    

        time_difference = get_time_difference(image_1, image_2) # Get time difference between images
        if time_difference !=0 :
            #print("timedif =!0")

#Here it is nessercery to make the condition that the timedifference is not zero

            image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects in black and white

    

    

    

    

            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors

            matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors

        #print(matches)

#        display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches

            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

        #print(coordinates_1[0], coordinates_2[0])

            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

        #print(average_feature_distance)

        #print(time_difference)

            speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)

            #print(speed)

    

            estimate_kmps = speed  # Replace with your estimate

    

        # Format the estimate_kmps to have a precision

        # of 5 significant figures

            estimate_kmps_formatted = "{:.4f}".format(estimate_kmps)

    

        # Create a string to write to the file

            output_string = speed#estimate_kmps_formatted

            #print(output_string)


            file_path = dir_path/"result.txt"

    

        # Write to the file

        #file_path = "result.txt"  # Replace with your desired file path
            #print(percentage1[2])
            percentcloud1=percentage1[3]
            #print(percentage2[2])
            percentcloud2=percentage2[3]
            percentground=percentage2[0]
            averagepercentcloud=(percentcloud1+percentcloud2)/2
            speedlist.append([output_string,averagepercentcloud])
        
            # Sorts by the second element
            speedlist = sorted(speedlist, key=lambda x: x[1])  
            print(speedlist)  
            #speedlist.sort()
            
            a=speedlist[0]
            print(a[0])
              
           
    spdlststr="\n".join(str(x) for x in speedlist)
    #print(spdlststr)
#    with open(file_path, 'a') as file:

#        file.write("Speed is the first number in each pair of brackets, cloudpercentage is the second.\n [Speed, Cloudpercentage] \n %s" % spdlststr)#speedlist)#(output_string,percentcloud1,percentcloud2,percentground, speedlist))

    
    estimate_kmps_formatted = "{:.4f}".format(a[0])

    with open(file_path, 'w') as file:
        file.write(estimate_kmps_formatted)
    print("Data written to", file_path)