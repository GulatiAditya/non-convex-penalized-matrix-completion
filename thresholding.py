import numpy as np
def threshold(S, thresh,_lambda, g):
    if (thresh== 'MCP'):
    	for i in range(len(S)):
    		if (S[i]>_lambda and S[i]<=g*_lambda):
    			S[i] = np.sign(S[i]*(abs(S[i])-_lambda)/(1-g**-1))
    		elif S[i]<_lambda:
    			S[i] = 0
    elif (thresh=='Soft'):
    	S = np.maximum(S-_lambda, 0)
    elif (thresh=="Hard"):
    	S = [x for x in S if x > (2*_lambda)**0.5]
    elif (thresh=="SCAD"):
    	for i in range(len(S)):
    		# print(S[i])
    		if(np.abs(S[i]) <= 2*_lambda):
    			# print(np.sign(S[i]))
    			# print(int(np.abs(S[i])-_lambda))
    			# print(max(0,int(np.abs(S[i])-_lambda)))
    			S[i] = np.sign(S[i])*max(0,int(np.abs(S[i])-_lambda))
    		elif(np.abs(S[i])>2*_lambda and np.abs(S[i])<= g*_lambda):

    			S[i] = ( (g-1)*S[i] - np.sign(S[i])*g*_lambda )/(g-2)
    return S

def penalty(S, pen, _lambda, g):
    if (pen=='Soft'):
        return (_lambda * (np.sum(S)))
    elif (pen=='MCP'):
        for i in range(len(S)):
            if(S[i]<=_lambda*g):
                S[i] = _lambda*S[i] - ((S[i]**2)/(2*g))
            elif (S[i]>_lambda*g):
                S[i] = 0.5 * g * _lambda**2
    elif (pen == "Hard"):
    	S = [1 for x in S if x!=0]
    elif (pen == "SCAD"):
    	return 0
    return np.sum(S)