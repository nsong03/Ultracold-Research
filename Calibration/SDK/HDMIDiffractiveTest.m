function HDMIDiffractiveTest()
%This function will loop through a series of images and load them to the
%SLM


% Camera initialize
clear
close all
% Load TLCamera DotNet assembly. The assembly .dll is assumed to be in the 
% same folder as the scripts.
NET.addAssembly([pwd, '\Thorlabs.TSI.TLCamera.dll']);
disp('Dot NET assembly loaded.');
tlCameraSDK = Thorlabs.TSI.TLCamera.TLCameraSDK.OpenTLCameraSDK;
% Get serial numbers of connected TLCameras.
serialNumbers = tlCameraSDK.DiscoverAvailableCameras;
disp([num2str(serialNumbers.Count), ' camera was discovered.']);
% Open the first TLCamera using the serial number.
disp('Opening the first camera')
tlCamera = tlCameraSDK.OpenCamera(serialNumbers.Item(0), false);
    
% Set exposure time and gain of the camera.
tlCamera.ExposureTime_us = 300;    
% Check if the camera supports setting "Gain"
gainRange = tlCamera.GainRange;
if (gainRange.Maximum > 0)
    tlCamera.Gain = 0;
end 
% Set the FIFO frame buffer size. Default size is 1.
tlCamera.MaximumNumberOfFramesToQueue = 5;
figure(1)
% Start continuous image acquisition
disp('Starting continuous image acquisition.');
tlCamera.OperationMode = Thorlabs.TSI.TLCameraInterfaces.OperationMode.SoftwareTriggered;
tlCamera.FramesPerTrigger_zeroForUnlimited = 0;
tlCamera.Arm;
tlCamera.IssueSoftwareTrigger;
maxPixelIntensity = double(2^tlCamera.BitDepth - 1);
% End of camera init




% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('ImageGen.dll', 'ImageGen.h');
end

% read the SLM height and width. 
width = calllib('Blink_C_wrapper', 'Get_Width');
height = calllib('Blink_C_wrapper', 'Get_Height');
depth = calllib('Blink_C_wrapper', 'Get_Depth');

% The number of data points we will use in the calibration is 256 (8 bit's)
NumDataPoints = 256;

% If you are generating a global calibration (recommended) the number of regions is 1, 
% if you are generating a regional calibration (typically not necessary) the number of regions is 64
NumRegions = 1;

%allocate an array for our image, and set the wavefront correction to 0 for the LUT calibration process
Image = libpointer('uint8Ptr', zeros(width*height*3,1));
WFC = libpointer('uint8Ptr', zeros(width*height*3,1));

% Create an array to hold measurements from the analog input (AI) board. 
% We ususally use a NI USB 6008 or equivalent analog input board.
AI_Intensities = zeros(NumDataPoints,2);

% When generating a calibration you want to use a linear LUT. If you are checking a calibration
% use your custom LUT 
if(height == 1152)
	lut_file = 'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\1920x1152_linearVoltage.lut';
else
	if(depth == 8)
		lut_file = 'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\19x12_8bit_linearVoltage.lut';
	end 
	if(depth == 10)
		lut_file = 'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\19x12_10bit_linearVoltage.lut';
	end
end

% Load the lookup table to the controller.  
calllib('Blink_C_wrapper', 'Load_lut', lut_file);
 
% Start with a blank pattern, and indicate that our images are RGB images
isEightBit = false;
RGB = true;
if (height == 1152)
	PixelValueOne = 0;
else
	PixelValueOne = 255;
end 
calllib('ImageGen', 'Generate_Solid', Image, WFC, width, height, depth, PixelValueOne, RGB);
calllib('Blink_C_wrapper', 'Write_image', Image, isEightBit);

PixelsPerStripe = 8;
%loop through each region
for Region = 0:(NumRegions-1)

    AI_Index = 1;
	%loop through each graylevel
	for Gray = 0:(NumDataPoints-1)
		if (height == 1152)
			PixelValueTwo = Gray;
		else
			PixelValueTwo = 255 - Gray;
		end 	
        %Generate the stripe pattern and mask out current region
        calllib('ImageGen', 'Generate_Stripe', Image, WFC, width, height, depth, PixelValueOne, PixelValueTwo, PixelsPerStripe, RGB);
        calllib('ImageGen', 'Mask_Image', Image, width, height, depth, Region, NumRegions, RGB);
            
        %write the image
        calllib('Blink_C_wrapper', 'Write_image', Image, isEightBit);
          
        %let the SLM settle for 40 ms (HDMI card can't load images faster than every 33 ms)
        pause(0.04);
            
        %YOU FILL IN HERE...FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
        numframes = 5;
        frameCount = 0;
        while frameCount < numframes
            if (tlCamera.NumberOfQueuedFrames >0)
                if (tlCamera.NumberOfQueuedFrames > 1)
                    disp(['You are slow.' num2str(tlCamera.NumberOfQueuedFrames) ' remains']);
                end
                imageFrame = tlCamera.GetPendingFrameOrNull;
                if ~isempty(imageFrame)
                    frameCount = frameCount +1;
                    imageData = uint16(imageFrame.ImageData.ImageData_monoOrBGR);
                    disp(['Image frame number: ' num2str(imageFrame.FrameNumber)]);

                    imageHeight = imageFrame.ImageData.Height_pixels;
                    imageWidth = imageFrame.ImageData.Width_pixels;
                    imageData2D = reshape(imageData, [imageWidth, imageHeight]);
                    figure(1),imagesc(imageData2D'), colormap(gray), colorbar
                end
                delete(imageFrame);
            end
            drawnow;
        end


        %SECOND: store the measurement in your AI_Intensities array
        AI_Intensities(AI_Index, 1) = Gray; %This is the difference between the reference and variable graysclae for the datapoint
        AI_Intensities(AI_Index, 2) = 0; % HERE YOU NEED TO REPLACE 0 with YOUR MEASURED VALUE FROM YOUR ANALOG INPUT BOARD

        AI_Index = AI_Index + 1;
    end
      
	% dump the AI measurements to a csv file
	filename = ['Raw' num2str(Region) '.csv'];
	csvwrite(filename, AI_Intensities);  
end

%Camerea de-init
disp('Stopping image acquisition.');
tlCamera.Disarm;
disp('Releasing camera');
tlCamera.Dispose;
delete(tlCamera);
delete(serialNumbers);
tlCameraSDK.Dispose;
delete(tlCameraSDK);


   
%blank the SLM again at the end of the test.
calllib('ImageGen', 'Generate_Solid', Image, WFC, width, height, depth, PixelValueOne, RGB);
calllib('Blink_C_wrapper', 'Write_image', Image, isEightBit);

if libisloaded('ImageGen')
    unloadlibrary('ImageGen');
end