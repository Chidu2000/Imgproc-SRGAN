classdef Discriminator < matlab.mixin.Copyable
    properties
        img_feat
        n_feats
        kernel_size
        act
        num_of_block
        patch_size
        conv01
        conv02
        body
        linear_size
        tail
    end
    
    methods
        function obj = Discriminator(img_feat, n_feats, kernel_size, act, num_of_block, patch_size)
            obj.img_feat = img_feat;
            obj.n_feats = n_feats;
            obj.kernel_size = kernel_size;
            obj.act = act;
            obj.num_of_block = num_of_block;
            obj.patch_size = patch_size;
            
            obj.conv01 = conv(in_channel, out_channel, kernel_size, false, act);
            obj.conv02 = conv(n_feats, n_feats, kernel_size, false, act, 2);
            
            body = cell(1, num_of_block);
            for i = 1:num_of_block
                body{i} = discrim_block(n_feats * (2 ^ (i - 1)), n_feats * (2 ^ i), kernel_size, act);
            end
            obj.body = nn.Sequential(body{:});
            
            obj.linear_size = ((patch_size / (2 ^ (num_of_block + 1))) ^ 2) * (n_feats * (2 ^ num_of_block));
            
            tail = cell(1, 4);
            tail{1} = nn.Linear(obj.linear_size, 1024);
            tail{2} = act;
            tail{3} = nn.Linear(1024, 1);
            tail{4} = nn.Sigmoid();
            obj.tail = nn.Sequential(tail{:});
        end
        
        function x = forward(obj, x)
            x = obj.conv01(x);
            x = obj.conv02(x);
            x = obj.body(x);
            x = reshape(x, [], obj.linear_size);
            x = obj.tail(x);
        end
    end
end
