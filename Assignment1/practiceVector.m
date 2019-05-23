test = [0:0.01:0.98]
test = test'
dat = ones(1,99).*randi(20,1,99)
dat = dat'

X = [test dat]
theta = [2.3;4.6]

theta = theta'
theta .* X