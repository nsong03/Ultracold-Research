function LoadImageSequence()
%This function will loop through a series of images and load them to the
%SLM

% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('ImageGen.dll', 'ImageGen.h');
end

%allocate arrays for our images
width = calllib('Blink_C_wrapper', 'Get_Width');
height = calllib('Blink_C_wrapper', 'Get_Height');
depth = calllib('Blink_C_wrapper', 'Get_Depth');
ImageOne = libpointer('uint8Ptr', zeros(width*height*3,1));
ImageTwo = libpointer('uint8Ptr', zeros(width*height*3,1));
%leave the wavefront correction blank. You should load your custom 
%WFC that was shipped with your SLM
WFC = libpointer('uint8Ptr', zeros(width*height*3,1));
    
% Generate a fresnel lens
CenterX = width/2;
CenterY = height/2;
Radius = height/2;
Power = 5;
cylindrical = true;
horizontal = true;
RGB = true;
calllib('ImageGen', 'Generate_FresnelLens', ImageOne, WFC, width, height, depth, CenterX, CenterY, Radius, Power, cylindrical, horizontal, RGB);

% Generate a blazed grating
Period = 128;
Increasing = 1;
horizontal = false;
calllib('ImageGen', 'Generate_Grating', ImageTwo, WFC, width, height, depth, Period, Increasing, horizontal, RGB);

% indicate that our images are RGB images
isEightBit = false;

% Loop between our images
for n = 1:10
	calllib('Blink_C_wrapper', 'Write_image', ImageOne, isEightBit);
	pause(1.0) % This is in seconds
 	calllib('Blink_C_wrapper', 'Write_image', ImageTwo, isEightBit);
 	pause(1.0) % This is in seconds
end

if libisloaded('ImageGen')
    unloadlibrary('ImageGen');
end