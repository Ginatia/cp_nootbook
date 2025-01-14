K种物品选出 N个物品，物品数无限制  

$$
x_1+x_2+ \cdots +x_k=n ,x_i\ge 0 \\
\binom{n+k-1}{n} \\
$$


若要求每个物品至少选择一种  

**处理物品出现次数的下界**

$$
x_1+x_2+ \cdots x_k=n,x_i\ge 1 \\
令 y_i= x_i -1 \\
y_1+y_2+ \cdots y_k=n-k,y_i \ge 0 \\
即通过变量替换化归问题\\
$$



**处理物品出现次数的上界**

*容斥原理*

对于多重集合 $T= \{3 \cdot a ,4 \cdot b , 5 \cdot c \} 的10组合的数目$  

$$
T^{*}=\{  \infty \cdot a,\infty \cdot b ,\infty \cdot c  \} 的10组合为： \\
|S|= \binom{10+3-1}{10}  \\
设 P_1: T^{*} 中 a 出现 > 3次 \\
P_2: T^{*} 中 a 出现 > 4次 \\
P_3: T^{*} 中 a 出现 > 5次 \\
A_i 为不具有 P_i 性质的集合大小  \\ 
原命题= |\overline{A_1 \cap A_2 \cap A_3}| = \\
|S|-(|A_1|+|A_2|+|A_3|)+(|A_1 \cap A_2|+|A_2 \cap A_3|+|A_1 \cap A_3|)-(|A_1 \cap A_2 \cap A_3|) \\
A_1 = 至少有 4个的情况 \\
A_2 = 至少有 5个的情况  \\
A_3 = 至少有 6个的情况  \\
划归于第一种情况，求下界即可 
$$













