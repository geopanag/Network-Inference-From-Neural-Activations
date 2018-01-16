#----------------- Transfer entropy from http://users.utu.fi/attenka/trent.R

# s, time shift
transferEntropy<-function(Y,X,s=1){
  
  #---------------------------------#
  # Transition probability vectors: #
  #---------------------------------#
  
  L4=L1=length(X)-s # Lengths of vector Xn+1.
  L3=L2=length(X) # Lengths of vector Xn (and Yn).
  
  #-------------------#
  # 1. p(Xn+s,Xn,Yn): #
  #-------------------#
  
  TPvector1=rep(0,L1) # Init.
  
  for(i in 1:L1)
  {
    TPvector1[i]=paste(c(X[i+s],"i",X[i],"i",Y[i]),collapse="") # "addresses"
  }
  
  TPvector1T=table(TPvector1)/length(TPvector1) # Table of probabilities.
  
  #-----------#
  # 2. p(Xn): #
  #-----------#
  
  TPvector2=X
  TPvector2T=table(X)/sum(table(X))
  
  #--------------#
  # 3. p(Xn,Yn): #
  #--------------#
  
  TPvector3=rep(0,L3)
  
  for(i in 1:L3)
  {
    TPvector3[i]=paste(c(X[i],"i",Y[i]),collapse="") # addresses
  }
  
  TPvector3T=table(TPvector3)/length(TPvector2)
  
  #----------------#
  # 4. p(Xn+s,Xn): #
  #----------------#
  
  TPvector4=rep(0,L4)
  
  for(i in 1:L4)
  {
    TPvector4[i]=paste(c(X[i+s],"i",X[i]),collapse="") # addresses
  }
  
  TPvector4T=table(TPvector4)/length(TPvector4)
  
  #--------------------------#
  # Transfer entropy T(Y->X) #
  #--------------------------#
  
  SUMvector=rep(0,length(TPvector1T))
  for(n in 1:length(TPvector1T))
  {
    SUMvector[n]=TPvector1T[n]*log10((TPvector1T[n]*TPvector2T[(unlist(strsplit(names(TPvector1T)[n],"i")))[2]])/(TPvector3T[paste((unlist(strsplit(names(TPvector1T)[n],"i")))[2],"i",(unlist(strsplit(names(TPvector1T)[n],"i")))[3],sep="",collapse="")]*TPvector4T[paste((unlist(strsplit(names(TPvector1T)[n],"i")))[1],"i",(unlist(strsplit(names(TPvector1T)[n],"i")))[2],sep="",collapse="")]))
  }
  return(sum(SUMvector))
}


t = dirname(parent.frame(2)$ofile)
setwd(t)
for(disc_meth in c("winning","chalearn"){
for(dat in 1:6){
  print("doing dataset:")
  print(dat)
  activations_loc = paste0("../data/discretized_",disc_meth,"_",dat,".csv")
  neurons = read.csv(activations_loc)
  neurons$X=NULL
  
  transfer_mat = matrix(nrow=ncol(neurons),ncol=ncol(neurons))
  
  for(i in 1:ncol(neurons)){
    for(j in 1:ncol(neurons)){
      if(i!=j){
	transfer_mat[i,j] = transferEntropy(neurons[,i],neurons[,j])
      }  
    }
  }
  
  write.csv(transfer_mat,paste0("../results/transfer_entropy_",disc_meth,"_",dat,".csv"))
}
}
