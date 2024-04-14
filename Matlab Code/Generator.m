
classdef Generator < matlab.mixin.Copyable
    properties
        img_feat
        n_feats
        kernel_size
        num_block
        act
        scale
        conv01
        body
        conv02
        tail
        last_conv
    end
    
    methods
        function obj = Generator(img_feat, n_feats, kernel_size, num_block, act, scale)
            obj.img_feat = img_feat;
            obj.n_feats = n_feats;
            obj.kernel_size = kernel_size;
            obj.num_block = num_block;
            obj.act = act;
            obj.scale = scale;
            
            obj.conv01 = conv(in_channel, out_channel, kernel_size, false, act);
            
            resblocks = cell(1, num_block);
            for i = 1:num_block
                resblocks{i} = ResBlock(n_feats, kernel_size, act);
            end
            obj.body = nn.Sequential(resblocks{:});
            
            obj.conv02 = conv(n_feats, n_feats, kernel_size, true, []);
            
            if scale == 4
                upsample_blocks = cell(1, 2);
                for i = 1:2
                    upsample_blocks{i} = Upsampler(n_feats, kernel_size, 2, act);
                end
            else
                upsample_blocks = cell(1, 1);
                upsample_blocks{1} = Upsampler(n_feats, kernel_size, scale, act);
            end
            obj.tail = nn.Sequential(upsample_blocks{:});
            
            obj.last_conv = conv(n_feats, img_feat, kernel_size, false, nn.Tanh());
        end
        
        function [x, feat] = forward(obj, x)
            x = obj.conv01(x);
            _skip_connection = x;
            
            x = obj.body(x);
            x = obj.conv02(x);
            feat = x + _skip_connection;
            
            x = obj.tail(feat);
            x = obj.last_conv(x);
        end
    end
end