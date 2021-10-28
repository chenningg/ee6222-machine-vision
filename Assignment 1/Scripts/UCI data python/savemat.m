clear
clc

dataset_path = 'C:\Users\Shi Qiushi\Desktop\RVFL_Python\UCI data(2) (1)\UCI data\';
[name,num] = GetFiles(dataset_path);
n_folders = num(1);
for i = 1:n_folders
    path = strcat(dataset_path,name(i).name,'\' );
    addpath(path);
    open_dataset(dataset_path, name(i).name);
    
    
end





function [] = open_dataset(dataset_path,dataset_name)
path = strcat(dataset_path,dataset_name,'\' );
[name_local,num_local] = GetFiles(path);
for i = 1:num_local(1)
    test = name_local(i).name;
    test(end-1:end)=[];
    savefile(test)
end
end

function [] = savefile(file_name)
eval(file_name);
try
    if size(index,1) == 1
        index = reshape(index, 8, 1);
    end
end
save(file_name)
end



function [names,class_num] = GetFiles(dataset_path)
files = dir(dataset_path);
size0 = size(files);
length = size0(1);
names = files(3:length);
class_num = size(names);
end