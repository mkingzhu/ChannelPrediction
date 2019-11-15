%% inputs: 
% mat_file_name: the input mat source file
% file_name_out: the output file that contains the channel information
% index: the channel index of the target channel that you want
%
% Example
% save_measured_mat_data('../data/source/AAPlantD1_2GHz_TX1_hpol_run4_pp.mat', '../data/measured/AAPlantD1_2GHz_TX1_hpol_run4_pp_1.dat', 1)
function save_measured_mat_data(mat_file_name, file_name_out, index)

load(mat_file_name, '-mat', 'IQdata');
data = IQdata(index, :);

data_file_out = fopen(file_name_out, 'w');

for l = 1:length(data)
    fwrite(data_file_out, real(data(l)), 'double', 'ieee-be');
    fwrite(data_file_out, imag(data(l)), 'double', 'ieee-be');
end

figure;
plot(real(data));
figure;
plot(imag(data));

end