function InitalizeSDK()
%This function will create a window used to load image data to the SLM, and
%will load a look up table to linearize the phase response of the SLM from
%0 to 2pi

% Load Blink_C_wrapper.dll
% should all be located in the same directory as the program referencing the library
loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h');

% Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
% the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (flase)
bCppOrPython = false;
calllib('Blink_C_wrapper', 'Create_SDK', bCppOrPython);
disp('Blink SDK was successfully constructed');

height = calllib('Blink_C_wrapper', 'Get_Height');
depth = calllib('Blink_C_wrapper', 'Get_Depth');

% Generate the path to the calibration of voltage to phase. The namse "linear.lut" is a bit confusing. 
% This linearly maps input voltages to output voltages, thus removing any calibration. 
% To be generic this example uses linear.lut, but you should link to your custom LUT delivered 
% with the SLM. 
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




