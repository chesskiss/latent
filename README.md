--- Long term


Hibernation mode - HOW TO AWAKEN/RETURN TO PROJECT:

1. Check artifical ring student initialization - Decoding from T->S on artificial ring models is worse than the teacher (T->T)
2. Make SGD work: 
- Consult with Kabir, and take his code entirely. 
- Start changing clean tutorial code or sgd_simplest_test
- Run a SHORT_RUN with VERBOSE in my code to check



Potential approaches:

I. Add more:
> students, especially bigger artifical rings
> perturbed teachers
> different randomly initalized teachers

II. Change run parameters on run on multi-teacher run (different teachers each), to find variability

III. Try with few shots instead of decoding? 

IV. Pertube the student and fit?

V. ?* Pertube the students and ... Try let the good models decode the bad, vice versa, and the bad decode the bad. Will "good" models decode the bad ones better than bad-> bad? Are there any conclusions?

VI. Try changing perturbation for an optimization one (like SGD)


Side tasks for better runs:
- Write a list of things I should double check to make sure everything is in order. Examples:
> likelihood on training data over time increasing, T likelihood on its train is best.
> Decoding from T->S on artificial ring models is worse than the teacher (T->T)
- ! Use GPT to save and load text (the hmm params values) in the correct format - To save models after T0.
- use vmap on decoding to save time and do multiple scores at once (start with teacher)



Goal - find a model that does well on T1, should also do reasonably well on T0 or vice versa?




---

Alternative tasks (archive):
. ! Plan/compute algo for generalizing students (s_1..1,s_1..2...?) for 5 teachers (T5) - combinatorics computation:
1->x->y where y>=x>=1
. Generalize teachers and students using the algorithms I developed and evaluate on an unseen teacher T6
. Visualize again w/ various dim reductions methods (PCA), or just likelihood, over T6.
. Try again for 99+1 teachers.
. try for several emission dims, and other generalizations + check specifications (final level)
*optional: 1 graph per 1 teacher's emissions, have 5 graphs total, or merge them (using different colors per teachers emissions eval)
. Check dynamax and Claude for how to pertub teachers properly (so the likelihood won't surpass the teacher? Or perhaps it is ok) VX - return to this step
?. Should I train the decoding again every time?


-------------------------------------------------------------------------------

What have I done/tried so far:
1.  Perturbed teachers?
2.  Trained, evaluated (likelihood), and compared students and teachers over teachers...
3.  Visualized initial true params emissions
    Evaluate all on unseen teacher T3 V
    Repeating each curriculum with many randomly initialized students. V
    Visualize results: Use rings on teacher's data, and graph the dataframe's data V
    Generalize student S_l V
    Keep function extractions and organizations for teachers, etc. V
    Fix 2D plotting V
4. Added decoding
5. Tried lower learning rate, and reducing/increasing epochs to analyse behavior

6. Reached high liklihood on students with decoding enabled.
7. Built a ring which starts better than random, but after it's likelihood is going up it goes down 
8. Create a good student - duplciate teacher, and add some states I cannot rach (sanity check that it works)
9. step 5 - to test for each epoch, rewrtie train(..) function : hmm_class.fit_em(params, prop, emissions, num_iters=1) , i.e. NUM_EPOCHS=1, and add a loop, and test each epoch 
10. Add deep copies of teachers (T0 most important) and train them over time on T0. Likelihood should always be 1.0 for T0 (the good model). 
That will fill 1 PAGE, 3x3 subpolots. 
Then, fit all the models (with the T0 teacher) on T2. We will have another PAGE of 3x3 graphs. 
-> Generalize*, for 2 teacher, have 2 copies of the model, so that we train them on different teachers.
11. Improve ring students initialization
12. Solve SGD on simple file
13. Update add_extranaous ring func to fit the new HMM. Use GPT

