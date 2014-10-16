function cnnnumgradcheck(net, x, y)
    epsilon = 1e-4;
    er      = 1e-8;
    n = numel(net.layers);

    for l = n : -1 : 2
        disp(['The value of l is ' num2str(l)]);
        for j = 1 : numel(net.layers{l}.a)
             disp(['The value of j is ' num2str(j)]);
                net_m = net; net_p = net;
                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                d = (net_p.L - net_m.L) / (2 * epsilon);
                e = abs(d - net.layers{l}.db{j})
                if e > er
                    error('numerical gradient checking failed');
                end
                for i = 1 : numel(net.layers{l - 1}.a)
                    disp(['The value of i is ' num2str(i)]);
                    for u = 1 : size(net.layers{l}.k{i}{j}, 1)
                        %u
                        for v = 1 : size(net.layers{l}.k{i}{j}, 2)
                            %v
                            net_m = net; net_p = net;
                            net_p.layers{l}.k{i}{j}(u, v) = net_p.layers{l}.k{i}{j}(u, v) + epsilon;
                            net_m.layers{l}.k{i}{j}(u, v) = net_m.layers{l}.k{i}{j}(u, v) - epsilon;
                            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                            d = (net_p.L - net_m.L) / (2 * epsilon);
                            e = abs(d - net.layers{l}.dk{i}{j}(u, v));
                            if e > er
                                error('numerical gradient checking failed');
                            end
                        end     
                    end
                end
        end
    end
%         if strcmp(net.layers{l}.type, 'c')
%             
%         elseif strcmp(net.layers{l}.type, 's')
%                for j = 1 : numel(net.layers{l}.a)
%                     net_m = net; net_p = net;
%                     net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
%                     net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
%                     net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                     net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                     d = (net_p.L - net_m.L) / (2 * epsilon);
%                     e = abs(d - net.layers{l}.db{j});
%                     if e > er
%                         error('numerical gradient checking failed');
%                     end
%                     for i = 1 : numel(net.layers{l - 1}.a)
%                         for u = 1 : size(net.layers{l}.k{i}{j}, 1)
%                             for v = 1 : size(net.layers{l}.k{i}{j}, 2)
%                                 net_m = net; net_p = net;
%                                 net_p.layers{l}.k{i}{j}(u, v) = net_p.layers{l}.k{i}{j}(u, v) + epsilon;
%                                 net_m.layers{l}.k{i}{j}(u, v) = net_m.layers{l}.k{i}{j}(u, v) - epsilon;
%                                 net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                                 net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                                 d = (net_p.L - net_m.L) / (2 * epsilon);
%                                 e = abs(d - net.layers{l}.dk{i}{j}(u, v));
%                                 if e > er
%                                     error('numerical gradient checking failed');
%                                 end
%                             end     
%                         end
%                     end
%                end
%         end
%     end
end
