function DestructSDK()
%This function will close the SDK. To properly close out the lower level
%code this will also close Matlab, so only call this function when you are
%done with your experiments.

calllib('Blink_C_wrapper', 'Delete_SDK');

unloadlibrary('Blink_C_wrapper');