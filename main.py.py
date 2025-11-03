"""
  Here comes first the code from the participants, then follows the pupils work with the code.
  That is the pupils commenting of the central parts of the code.
  The pupils have tested an discussed changes and improvements to the code in collaboration with the teacher.
  The results of the code is saved in the file named 'results.txt'
  The focus here has been the correlation between the calculation of the speed and the percentage of clouds
  in the images taken from the ISS.
  So that is why the results file is filled with collumns one with the calculated speed
  and the next one with the calculated percantage of clouds in the two images used for the speed calculation.  
  x : matrix of x positions
  
  Fra linje 1-4 er det initialisering som et startopsætningen af programmet. Den henter og importerer
de biblioteker, f.eks. biblioteket ”image” i linje 1, biblioteket ”datetime” i linje 2, biblioteket ”cv2” i
linje 3 og til sidst biblioteket ”math” i linje 4.
Den første funktion der er defineret er ”Get_time”. Alt det funktionen har nedenunder sig indtil
”return” er funktionens evne. Her åbner den et ”image” og henter det i linje 8. I billedet har den en
variabel ”datetime_original” (linje 12), og dertil laver den det om til et andet format så det er
nemmere at læse variablen. Hvor den til sidst i linje 14 returnerer ”time”.
Pamilla Janphongam & Olivia Klitgaard Informatik C
Brøndby Gymnasium 3.i
I denne del af kodningen har vi 5 forskellige funktioner, som vi hver for sig vil forklare.
Funktionen ”convert_to_cv” (linje 28) er den funktion som gør at billederne konverteres til formatet
cv. Her er det både ”image_1” og image_2” er bliver omdannet til cv. Når den har ændret formatet
retunere den billederne.
I funktionen ”calculate_features”(linje 32) finder den kendetegn ved det nyes formats billeder.
Features er nemlig kendetegn så derfor regner (calculate) den frem til kendetegene på billederne.
Man kan se i linje 34 og 35 at den gør det for begge billeder. Det kan f.eks. være de lyseste punkter
eller en stærk kontrast. Den beregner og finder altså antallet af features og retunere dem til sidst i
funktionen.
Den definerede funktion ”calculate_matches”(linje 37) er her hvor den leder efter billeder der ligner
hinanden. Dette gør den ved at bruge brute force. Den tager alle features og tjekekr igennem ved
hjælp af brute force som kører igennem alle billederne, dertil matcher den dem der er ens eller
ligner hinanden. Man kan se i linje 40 at den sorterer de match den har fundet, hvorved den retunere
de matches den fandt ved brug af brute force og featuresene i billederne.
Da vi havde defineret disse 3 funktioner, skulle de prøves ifølge vejledningen. De 3 funktioner
konvertering (convert) til computer vision (cv), finder features, tage featuresene og fører dem ind
som argumenter til at se om der er macthes, og hvis der var retunerer dem.

I funktionen ”find_matches_coorinatats” (linje 48) matcher den koordinaterne 1 og 2. Den har to
koordinater (1 og 2) hvor ved den bruger match in matches. Den tager ”image_1” og ”image_” da
den ved ud fra featuresene hvad der er macthes. Dertil bruger den x1 og y1 til at returnerer
koordinat 1 og 2.

Funktionen ”calculate_mean_distance” er hvor den kører alle koordinaterne igennem og finder
genenmsnitsafstanden ved brug af x-forskelle og y-forskelle, hvorved den finder alle distancer, ved
at tage alle distancer og plusse med distance. Den tager længden af distancer ved bruger af Merged
som er det samme som len = længde. Den tager alle distancer og dividere med længden af summen
for distancer. Inden i funktionen har vi ” for coordinate in merged_coordinates” som ses i linje 62.
Så længe det er opfyldt kan det beregne afstanden. Det er nemlig en liste over afstande hvorved den
returnerer afstandene igen. Derefter printer koordinaterne, både 1 og 2. Dette kalder man en løkke,
en betinget løkke. Det gør man fordi den bliver ved at gøre det samme indtil den har være gennem
dem alle

I denne funktion beregner den distancen ved at den tage ”feauture_distance”, ganger det med GSD (pixel
tilkilometer) og dividerer med 100000. Dette bruger den til at beregne ”speed”, ved at den nu i linje 72 har
distance som den også bruger i linje 73. Her regner den nemlig speed ved at tage resultatet at ”distance” og
dividere det med tidsforskellen ”time_difference”. Dertil returnerer den ”speed”.

I denne del af koden har den to billeder ”image 1, image 2”. Den starter med at finde tidsforskellen
mellem de to billeder i linje 79. Dertil konverterer den til formatet cv i linje 80. I linje 81 beregner
den ”features” altså kendetegnene, hvorved den finder frem til matchene og udplukker de bedste. I
linje 85 finder den matchende koordinater for at finde ud af hvor meget de to billeder har rukket dig
fra hinanden.
I linje 87 regner den ”mean_distance” for koordinat 1 og 2. Til sidst regner den speed i linje 89 ved
at beregne speed in kmps. Den har nemlig (average_feauture_distance, 12648 som er den beregnede
distance og til sidst tidsforskellen).

Dette er den sidste del af koden, hvor den opretter en fil som hedder ”result.txt” som kan ses i linje
102. Her sætter den vores beregnede hastighed ind.
I mappen hvor billeder og Python filen ligger,kommer der et dokument som ses under.

I linje 61
kører funktionen, hvor den klassificerer billedet og viser det. Fordi som vi ser i funktionen oven
over der kører fra linje 48-linje 59. Det starter i linje 52 med at sige hvis den skal vise billeder, skal
den først finde jordoverfladen, finde skyerne og finde havde. Det er nogle funktioner længere oppe i
koden vi kommer til.

I koden med funktionen defineret kaldet ”calculate_percentage” er her hvor den tæller antal pixler.
Man ser f.eks. i linje 41 beregner den antal pixler for border, i linje 2 beregner den antal pixler for
havet, i linje 43 er det antal pixler for skyerne og til sidst i linje 44 beregner den antal pixler for
jordoverfladen. Dertil finder den summen af det totale antal pixler, som man kan se, bliver beregnet
i linje 45. Dog kan det med antal pixler for hver af delelementerne være lidt misvisende. I linje 46
returnerer den et regnestykke. Den tager det fundet antal pixler der ligger i grænserne og dividere
med summen af antal pixler i alt. Antal pixler i skyernes billeder og dividerer med hele antal af
pixlerne, dertil antal landjords pixler divideres med alles antal pixler og minus med grænserne
pixler.

Funktionen oven over kaldet ”vis_billede”, er her hvor den viser billeder.
Den nederste funktion ”compute_percentage” har vi i linje 30 et masked billede (masked = Det er et
billede hvor der står et 1 tal hvert sted der er noget vi gerne vil have ud eller fjernet på billedet)
I linje 31 er her hvor den laver det om, så vi får en udgave af pixel, både korte og lange. Den tager
de pixler der er masket ud. I linje 32 ligger den alle pixlerne sammen så den får en pixel sum
hvorved den i linje 32 får det antal pixler som kunne være for jord, hav eller andet som den så
returnerer.
I funktionen ”find_ground” laver den en xor funktion, som er en logsik funktion. Det gør den for at
finde jorden, og der vil den gerne hvade det der ikke er grænserne, havet og skyer så derfor bruger
den en xor funktion. Den udelukker nemlig de der ikke ønskes og dertil kan den i linje 28 returnerer jorden

I disse funktioner vil vi gerne finde skyerne og havet, fordi det brugte vi jo før i dette program.
Der sker det samme i begge funktioner bare med forskellige tærskler, men hvis man tager den med
havet. I linje 13 bliver billedet lavet om til et hsv. I linje 14 kigger den på hsv billedet, også tager
den tærskler (Det er tærskelværdier, dem hvor man kan hive og justerer). Der er nemlig en max og
en min. I linje 15 laves der en maske ud fra tærsklerne. Mask er nemlig en maske der gør at vi får et
1 tal hvor noget skal klippes ud. Det er bare et hvidt sort billede. Det kan vi bruge til at få vist det
billede hvor de faktisk pixel værdier er. Til sidst i linje 16 returnerer den det billede ud, med kun de
pixelværdier som skal ses. Det er f.eks. ved at fjerne skyerne fra havet, så det kan ses ordentlig.
Hele denne samme proces sker for funktionen ”find_cloud”. Her fjerne den bare pixelværdierne så
der kun kan ses skyer som også returneres som output til sidst.

I funktionen ”find_border” sker det samme som i ”find_cloud”, ”find_sea”og ”find_ground”. Her
finder den bare grænsen, som er det sorte rundt om koøjet. Dette gør den som de 3 andre funktioner
blot med andre tærskelværdier.
I linje 4 kan det ses at det er et billede som er importeret. Det er nemlig Bangkok billedet, som man
smider ind og bruger som argument. Dette skal nemlig bruges for at finde funktionerne
I linje 1 og 2 importeres bibliotekerne ”numpy” og ”cv2” som værende inistialisering, så
programmet kan fungere.

In this passage we have described it in danish.

Dette er den sidste del af koden, hvor den opretter en fil som hedder ”result.txt” som kan ses i linje
102. Her sætter den vores beregnede hastighed ind.

It is in the last part of the code, that we got a file named ”result.txt”. In the file we got the speed.
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