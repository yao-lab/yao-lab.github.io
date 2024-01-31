d = 20;
reps = 50;
success = zeros(d);
for n = 1:d
    disp(n);
    for k = 1:d
        disp(k);
        success_count = 0;
        for j = 1:reps
%             Construct a spare vector
            x0 = zeros(d,1);
            idx = randsample(d,k);
            values = randsample([-1,1], k, true);
            x0(idx) = values;
%             Create a Gaussian random matrix
            A = randn(n, d);
            b = A * x0;
%             Solve the problem
            cvx_solver SeDuMi
            cvx_begin quiet
                variables x_des(d);
                minimize(sum(abs(x_des)));
                subject to
                    (A * x_des) == b;
            cvx_end
%             Declare success
            if ~contains(["Infeasible", "Unbounded"], cvx_status)
                if norm(x_des - x0) <= 1e-3
                    success_count = success_count + 1;
                end
            end
        end
        success(n,k) = success_count / reps;
    end
end

%%
fig = imagesc(success);
exportgraphics(gca, "prob.png", Resolution=150);
