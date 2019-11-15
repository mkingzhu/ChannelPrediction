%% inputs: 
% csv_file_name: the input csv source file
% file_name_out: the output file that contains the channel information
%
% Example
% save_measured_csv_data('Loc_0109_Lab_139_6Ch1.csv', '../data/measured/Loc_0109_Lab_139_6Ch1.dat')
function save_measured_csv_data(csv_file_name, file_name_out)

data = csvread(csv_file_name);

data_file_out = fopen(file_name_out, 'w');

for l = 1:length(data)
    fwrite(data_file_out, data(l, 4), 'double', 'ieee-be');
    fwrite(data_file_out, data(l, 5), 'double', 'ieee-be');
end

figure;
plot(data(:, 4));
figure;
plot(data(:, 5));

end