classdef mydata < matlab.mixin.Copyable
    properties
        LR_path
        GT_path
        in_memory
        transform
        LR_img
        GT_img
    end
    
    methods
        function obj = mydata(LR_path, GT_path, in_memory, transform)
            obj.LR_path = LR_path;
            obj.GT_path = GT_path;
            obj.in_memory = in_memory;
            obj.transform = transform;
            
            obj.LR_img = dir(fullfile(LR_path, '*.jpg'));
            obj.GT_img = dir(fullfile(GT_path, '*.jpg'));
            
            if in_memory
                for i = 1:length(obj.LR_img)
                    obj.LR_img(i).data = imread(fullfile(obj.LR_path, obj.LR_img(i).name));
                    obj.GT_img(i).data = imread(fullfile(obj.GT_path, obj.GT_img(i).name));
                end
            end
        end
        
        function len = length(obj)
            len = length(obj.LR_img);
        end
        
        function img_item = getitem(obj, i)
            img_item = struct();
            
            if obj.in_memory
                GT = im2single(obj.GT_img(i).data);
                LR = im2single(obj.LR_img(i).data);
            else
                GT = im2single(imread(fullfile(obj.GT_path, obj.GT_img(i).name)));
                LR = im2single(imread(fullfile(obj.LR_path, obj.LR_img(i).name)));
            end
            
            img_item.GT = (GT / 127.5) - 1.0;
            img_item.LR = (LR / 127.5) - 1.0;
            
            if ~isempty(obj.transform)
                img_item = obj.transform(img_item);
            end
            
            img_item.GT = permute(img_item.GT, [3, 1, 2]);
            img_item.LR = permute(img_item.LR, [3, 1, 2]);
        end
    end
end