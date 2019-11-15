%% inputs: 
% file_name_in: the source file
% file_name_out: the target file
% length: the length of the channel
% SNR: the noise level
%
% Example
% noise_generator('../data/measured/Loc_0109_Lab_139_6Ch1.dat', '../data/measured/SNR20_Loc_0109_Lab_139_6Ch1.dat', 600, 20)
function noise_generator(file_name_in, file_name_out, length, SNR)

real_part = zeros(length, 1);
imag_part = zeros(length, 1);

data_file = fopen(file_name_in, 'r');
for i = 1:length
    real_part(i, 1) = fread(data_file, 1, 'double', 'ieee-be');
    imag_part(i, 1) = fread(data_file, 1, 'double', 'ieee-be');
end
fclose(data_file);

h = complex(real_part, imag_part);
hh = awgn(h, SNR, 'measured', 'db');

real_part = real(hh(:, 1));
imag_part = imag(hh(:, 1));

data_file = fopen(file_name_out, 'w');
for i = 1:size(real_part, 1)
    fwrite(data_file, real_part(i, 1), 'double', 'ieee-be');
    fwrite(data_file, imag_part(i, 1), 'double', 'ieee-be');
end
fclose(data_file);

