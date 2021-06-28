

function y = validate(fIn_test)

load 'gold_standard.mat' fIn; % get fIn

y = norm(fIn_test - fIn,2);



