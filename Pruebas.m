%% TRAINING CODE
clear,clc
%% Obtaining all the photos
imds = imageDatastore('fotos', 'LabelSource', 'foldernames', 'IncludeSubfolders',true); %With this command, we obtain an image datastore
%and give each photo an individual label that depends on which browse
%directory it is. This will help us optimizing the memory because the photos
%are only in the workspace when you call them. It helps too with the
%organization of the datastore.
tibu = find(imds.Labels == 'tibu'); %Here we separate the photos based on their label
ys = find(imds.Labels == 'ys'); %so we have the images of each subject
%% Setting the weights, biases, targets and learning rate
R = 59; %Each input will have 59 values that correspond to each photo caracteristics
N1=8; %We'll use 8 neurons in the input layer
N2=1; %And only one on the output, because we only want to recognize between 2 diferent subjects
w1=rand(N1,R);
b1=rand(N1,1);
w2=rand(N2,N1);
b2=rand(N2,1);
alpha=0.001; %We set a learning rate of 0.001
t=[1 -1]; % And our targets will be 1 for subject one and -1 for subject two
%% Here comes the BackPropagation!
p = zeros(59,2); %We initialize the patterns like a matrix of zeros
for i = 1:23 %We have 23 training images for each subject and we'll use them all for training process
    facet = readimage(imds,tibu(i)); %We get to the workspace the 'i' image of subject one
    facet = rgb2gray(facet); %And then we turn it from RGB to grayscale
    lbpt = extractLBPFeatures(facet); %We extract subject's one image Local Binary Pattern features
    facey = readimage(imds,ys(i)); %We get to the workspace the 'i' image of subject two
    facey = rgb2gray(facey); %And then we turn it from RGB to grayscale
    lbpy = extractLBPFeatures(facey); %We extract subject's two image Local Binary Pattern features
    for j = 1:59 %We fill the patterns vector
        p(j, :) = [lbpt(j), lbpy(j)]; %using both subject's extracted LBP features
    end
    
    for rep=1:9999999 %We set a lot of epochs
        for k=1:2 %And we use both subject's features
            a1=logsig(w1*p(:,k)+b1); %We propagate
            a2=purelin(w2*a1+b2);    %the input forward
            e=t(k)-a2; %And then we calculate the error
            S2=-2*e; %After that we calculate the sensitivities so we can
            S1=[(1-a1(1))*a1(1) 0 0 0 0 0 0 0; %know each layer contribution
                0 (1-a1(2))*a1(2) 0 0 0 0 0 0; %to the total error
                0 0 (1-a1(3))*a1(3) 0 0 0 0 0;
                0 0 0 (1-a1(4))*a1(4) 0 0 0 0;
                0 0 0 0 (1-a1(5))*a1(5) 0 0 0;
                0 0 0 0 0 (1-a1(6))*a1(6) 0 0;
                0 0 0 0 0 0 (1-a1(7))*a1(7) 0;
                0 0 0 0 0 0 0 (1-a1(8))*a1(8)]*w2'*S2;
            w2=w2-alpha*S2*a1'; %Finally, we
            b2=b2-alpha*S2; %recalculate the new
            w1=w1-alpha*S1*p(:,k)'; %weights
            b1=b1-alpha*S1; %and biases
        end
    end
    
end
%% Saving the results
save results.mat w1 w2 b1 b2 %When all the training process has been done, we save the final weights and biases in a .mat file