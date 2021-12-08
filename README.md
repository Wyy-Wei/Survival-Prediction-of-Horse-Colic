# Survival-Prediction-of-Horse-Colic


## Introduction

Colic in horses, simply refering to abdominal pain, is one of the most common problems in equine practice. It has a significant economic impact on the racehorse industry and is a major concern for owners.

Equine colic can be divided into 2 major categories; gastrointestinal and nongastrointestinal. Horse’s vital signs (heart rate, respiratory rate, and mucus membrane color) is important information for diagnosis. Nongastrointestinal colic cases can usually be excluded based on physical examination findings; these include signs of abdominal discomfort due to urinary urolithiasis and disorders of reproductive, nervous, respiratory, or musculoskeletal systems. Causes of gastrointestinal colic (GC) are gut distension, tension on the root of mesentery, ischemia, deep ulcers in the stomach or bowel, and peritoneal pain\cite{colic}.

The purpose of this study was to explore the symptoms of horses with colic and predict survival, with a view to providing equine practitioners with a better reference to equine colic condition, which may contribute to timely treatment and higher colic survival rates.

## Data Description

The pathology data of 368 horses presented with signs of colic is reviewed. It includes 28 attributes about the vital signs and surgical information of horses including rectal temperature, mucous membranes, nasogastric reflux, etc. The attributes can be continuous, discrete, and nominal. 

\begin{table}[t!]
\caption{Attribute Information}
\begin{center}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Variable}&\textbf{Type}&\textbf{Attribute Information}\\
\hline
surgery & categorical & with or without surgery \\
\hline
age & categorical & 1 = Adult horse, 2 = Young \\
\hline
hospital\_num & character & the case number assigned to the horse\\
\hline
rectal\_temp & numerical & rectal temperature \\
\hline
pulse & numerical & the heart rate in beats per minute\\
\hline
respi\_rate & numerical & respiratory rate\\
\hline
t\_extremities & ordinal & indication of peripheral circulation\\
\hline
peri\_pulse & ordinal & peripheral pulse, subjective\\
\hline
color & categorical &  a subjective measurement of colour\\
\hline
refill\_time & categorical & capillary refill time\\
\hline
pain & categorical & the horse's pain level\\
\hline
peristalsis & ordinal & activity in the horse's gut\\
\hline
abd\_distend & ordinal  & abdominal distension\\
\hline
gas\_tube & ordinal &  any gas coming out of the tube\\
\hline
gas\_reflux & categorical & nasogastric reflux\\
\hline
reflux\_ph & numerical & nasogastric reflux PH\\
\hline
feces & ordinal &  rectal examination of feces\\
\hline
abdomen & categorical  & state of intestines\\
\hline
cell\_vol & numerical & number of red cells in the blood\\
\hline
total\_protein & numerical & normal in 6-7.5 (gms/dL) \\
\hline
abd\_appea & categorical & fluid obtained from the abdominal cavity\\
\hline
abd\_protein & numerical & abdomcentesis total protein \\
\hline
outcome & categorical & lived, died or was euthanized\\
\hline
surgical\_les & categorical &  was the lesion surgical\\
\hline
lesion1, 2, 3 & character & type of lesion\\
\hline
cp & categorical & is pathology data present \\
\hline
\end{tabular}
\label{data}
\end{center}
\end{table}
