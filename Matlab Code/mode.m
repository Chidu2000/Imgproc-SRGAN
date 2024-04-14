function mode(args)
    % Define cropping function
    function cropped_image = crop_image(image, scale, patch_size)
        [h, w, ~] = size(image);
        crop_h = patch_size * scale;
        crop_w = patch_size * scale;
        x = randi([1, h - crop_h + 1]);
        y = randi([1, w - crop_w + 1]);
        cropped_image = image(x:x+crop_h-1, y:y+crop_w-1, :);
    end

    % Define augmentation function
    function augmented_image = augment_image(image)
        % Example: apply horizontal flip with 50% probability
        if rand() > 0.5
            augmented_image = flip(image, 2);
        else
            augmented_image = image;
        end
        % You can include more augmentation techniques here
    end

    % Define transformation pipeline
    function transformed_image = transform_image(image)
        cropped_image = crop_image(image, args.scale, args.patch_size);
        transformed_image = augment_image(cropped_image);
    end

    
    % Define transformations
    transform = {@transform_image};
    
    % Create dataset
    dataset = mydata(args.GT_path, args.LR_path, args.in_memory, transform);
    
    % Create DataLoader
    loader = createDataLoader(dataset, args.batch_size, true, args.num_workers);
    
    % Create generator
    generator = Generator('img_feat', 3, 'n_feats', 64, 'kernel_size', 3, 'num_block', args.res_num, 'scale', args.scale);
    
    % Load pre-trained model if fine-tuning
    if args.fine_tuning
        generator.load_state_dict(torch.load(args.generator_path));
        disp("pre-trained model is loaded");
        disp("path : " + args.generator_path);
    end
    
    % Move generator to device and set to train mode
    generator = generator.to(device);
    generator.train();
    
    % Define loss and optimizer for generator
    l2_loss = nn.MSELoss();
    g_optim = optim.Adam(generator.parameters(), 'lr', 1e-4);
    
    % Initialize epoch counters
    pre_epoch = 0;
    fine_epoch = 0;
    
    % Pre-training loop
    while pre_epoch < args.pre_train_epoch
        for i = 1:numel(loader)
            tr_data = loader{i};
            gt = tr_data('GT').to(device);
            lr = tr_data('LR').to(device);
            output = generator(lr);
            loss = l2_loss(output, gt);
            g_optim.zero_grad();
            loss.backward();
            g_optim.step();
            
            pre_epoch = pre_epoch + 1;
            if mod(pre_epoch, 2) == 0
                disp(pre_epoch);
                disp(loss.item());
                disp('=========');
            end

            if mod(pre_epoch, 800) == 0
                torch.save(generator.state_dict(), sprintf('./model/pre_trained_model_%03d.pt', pre_epoch));
            end
        end
    end
    
    % Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device);
    vgg_net = vgg_net.eval();
    
    discriminator = Discriminator('patch_size', args.patch_size * args.scale);
    discriminator = discriminator.to(device);
    discriminator.train();
    
    d_optim = optim.Adam(discriminator.parameters(), 'lr', 1e-4);
    scheduler = optim.lr_scheduler.StepLR(g_optim, 'step_size', 2000, 'gamma', 0.1);
    
    VGG_loss = perceptual_loss(vgg_net);
    cross_ent = nn.BCELoss();
    tv_loss = TVLoss();
    real_label = torch.ones(args.batch_size, 1).to(device);
    fake_label = torch.zeros(args.batch_size, 1).to(device);
    
    while fine_epoch < args.fine_train_epoch
        scheduler.step();
        
        for i = 1:numel(loader)
            tr_data = loader{i};
            gt = tr_data('GT').to(device);
            lr = tr_data('LR').to(device);
            
            % Training Discriminator
            output = generator(lr);
            fake_prob = discriminator(output);
            real_prob = discriminator(gt);
            
            d_loss_real = cross_ent(real_prob, real_label);
            d_loss_fake = cross_ent(fake_prob, fake_label);
            
            d_loss = d_loss_real + d_loss_fake;
            g_optim.zero_grad();
            d_optim.zero_grad();
            d_loss.backward();
            d_optim.step();
            
            % Training Generator
            output = generator(lr);
            fake_prob = discriminator(output);
            
            [percep_loss, hr_feat, sr_feat] = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, 'layer', args.feat_layer);
            
            L2_loss = l2_loss(output, gt);
            percep_loss = args.vgg_rescale_coeff * percep_loss;
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label);
            total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat).^2);
            
            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss;
            
            g_optim.zero_grad();
            d_optim.zero_grad();
            g_loss.backward();
            g_optim.step();
        end
        
        fine_epoch = fine_epoch + 1;
        if mod(fine_epoch, 2) == 0
            disp(fine_epoch);
            disp(g_loss.item());
            disp(d_loss.item());
            disp('=========');
        end
        
        if mod(fine_epoch, 500) == 0
            torch.save(generator.state_dict(), sprintf('./model/SRGAN_gene_%03d.pt', fine_epoch));
            torch.save(discriminator.state_dict(), sprintf('./model/SRGAN_discrim_%03d.pt', fine_epoch));
        end
    end
end

