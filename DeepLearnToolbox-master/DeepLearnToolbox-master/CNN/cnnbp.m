function net = cnnbp(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2*sum(net.e(:) .^ 2) / size(net.e, 3);

    %%  backprop deltas
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} =  net.e;   %  output delta
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                    z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z.*(net.layers{l}.a{i}).*(1 - net.layers{l}.a{i});
               % net.layers{l}.d{i} = z.*double(net.layers{l}.a{i}>0);
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        elseif strcmp(net.layers{l}.type, 'o')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end    
        end
    end
 

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
