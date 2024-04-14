% Read the images
lowResImage = imread('<Low res Image Path>');
superResImage = imread('<High res Image Path>'); % Assuming Python output is PNG

% Check if high-resolution image exists
if exist('G:/My Documents/MATLAB/SRGAN_matlab/data_set/HR/res_0001.PNG', 'file')
  highResImage = imread('G:/My Documents/MATLAB/SRGAN_matlab/data_set/HR/res_0001.PNG');
else
  % Handle case where high-resolution image is not available
  disp('High-resolution image not found. Only low-resolution and super-resolved images will be compared.');
  highResImage = [];
end

% Define a subplot layout for 2 or 3 images
if isempty(highResImage)
  numRows = 1;
  numCols = 2;
else
  numRows = 1;
  numCols = 3;
end

% Create the figure
figure('Name', 'Image Comparison');

% Subplot for low-resolution image
subplot(numRows, numCols, 1);
imshow(lowResImage);
title('Low Resolution');

% Subplot for super-resolution image
subplot(numRows, numCols, 2);
imshow(superResImage);
title('Super-Resolved (Python)');

% Subplot for high-resolution image (if available)
if ~isempty(highResImage)
  subplot(numRows, numCols, 3);
  imshow(highResImage);
  title('High Resolution');
end