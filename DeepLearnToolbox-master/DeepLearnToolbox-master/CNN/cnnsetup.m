function net = cnnsetup(net, x, y)
    inputmaps = net.layers{1}.inputmaps;
    mapsize = size(squeeze(x(:, :, 1, 1)));
    for l = 1 : numel(net.layers)   
        if strcmp(net.layers{l}.type, 'o')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = normrnd(0,0.001,net.layers{l}.kernelsize,net.layers{l}.kernelsize);
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
            
        end
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = normrnd(0,0.001,net.layers{l}.kernelsize,net.layers{l}.kernelsize);
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
   
end
