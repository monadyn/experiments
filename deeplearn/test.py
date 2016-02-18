e
n = p_1 > 0.5                    # The prediction thresholded
        #xent =  # Cross-entropy loss function
        #-(T.sum( y*T.log(net) + (1-y)*T.log(1-net) ) ) 
        #cost = -(T.sum(y * T.log(p_1) + (1-y) * T.log(1-p_1)) )          # The cost to minimize
        #gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost    
        #cost func
        #cost(hθ, (x),y) = -ylog( hθ(x) ) - (1-y)log( 1- hθ(x) ) 
        #self.output = -(T.sum( y*T.log(net) + (1-y)*T.log(1-net) ) )


        #net = T.dot(input, self.W) + self.b
        # Construct Theano expression graph
        p_1 = 1 / (1 + T.exp(-T.dot(input, self.W  - self.b))   # Probability that target = 1
