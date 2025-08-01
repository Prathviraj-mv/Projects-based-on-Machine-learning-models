import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics   import  classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv('advertising.csv')


data.drop(columns='City')
data.drop(columns='Country')
data.drop(columns='Ad Topic Line')
data.drop(columns='Timestamp')
print(data.columns)
print(data.head())
# sns.histplot(x='Age',y ='Area Income',data =data)
# sns.pairplot(data)
# sns.jointplot(x='Age',y ='Daily Time Spent on Site',data =data,kind ='kde',color='red')
# sns.jointplot(x='Age',y ='Daily Internet Usage',data =data,color='red')

sns.pairplot(data,hue='Clicked on Ad')

X= data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y =data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

LOR =LogisticRegression()
LOR.fit(X_train,y_train)
pred = LOR.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))




plt.show()



Daily Time Spent on Site,Age,Area Income,Daily Internet Usage,Ad Topic Line,City,Male,Country,Timestamp,Clicked on Ad
68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11,0
80.23,31,68441.85,193.77,Monitored national standardization,West Jodi,1,Nauru,2016-04-04 01:39:02,0
69.47,26,59785.94,236.5,Organic bottom-line service-desk,Davidton,0,San Marino,2016-03-13 20:35:42,0
74.15,29,54806.18,245.89,Triple-buffered reciprocal time-frame,West Terrifurt,1,Italy,2016-01-10 02:31:19,0
68.37,35,73889.99,225.58,Robust logistical utilization,South Manuel,0,Iceland,2016-06-03 03:36:18,0
59.99,23,59761.56,226.74,Sharable client-driven software,Jamieberg,1,Norway,2016-05-19 14:30:17,0
88.91,33,53852.85,208.36,Enhanced dedicated support,Brandonstad,0,Myanmar,2016-01-28 20:59:32,0
66.0,48,24593.33,131.76,Reactive local challenge,Port Jefferybury,1,Australia,2016-03-07 01:40:15,1
74.53,30,68862.0,221.51,Configurable coherent function,West Colin,1,Grenada,2016-04-18 09:33:42,0
69.88,20,55642.32,183.82,Mandatory homogeneous architecture,Ramirezton,1,Ghana,2016-07-11 01:42:51,0
47.64,49,45632.51,122.02,Centralized neutral neural-net,West Brandonton,0,Qatar,2016-03-16 20:19:01,1
83.07,37,62491.01,230.87,Team-oriented grid-enabled Local Area Network,East Theresashire,1,Burundi,2016-05-08 08:10:10,0
69.57,48,51636.92,113.12,Centralized content-based focus group,West Katiefurt,1,Egypt,2016-06-03 01:14:41,1
79.52,24,51739.63,214.23,Synergistic fresh-thinking array,North Tara,0,Bosnia and Herzegovina,2016-04-20 21:49:22,0
42.95,33,30976.0,143.56,Grass-roots coherent extranet,West William,0,Barbados,2016-03-24 09:31:49,1
63.45,23,52182.23,140.64,Persistent demand-driven interface,New Travistown,1,Spain,2016-03-09 03:41:30,1
55.39,37,23936.86,129.41,Customizable multi-tasking website,West Dylanberg,0,Palestinian Territory,2016-01-30 19:20:41,1
82.03,41,71511.08,187.53,Intuitive dynamic attitude,Pruittmouth,0,Afghanistan,2016-05-02 07:00:58,0
54.7,36,31087.54,118.39,Grass-roots solution-oriented conglomeration,Jessicastad,1,British Indian Ocean Territory (Chagos Archipelago),2016-02-13 07:53:55,1
74.58,40,23821.72,135.51,Advanced 24/7 productivity,Millertown,1,Russian Federation,2016-02-27 04:43:07,1
77.22,30,64802.33,224.44,Object-based reciprocal knowledgebase,Port Jacqueline,1,Cameroon,2016-01-05 07:52:48,0
84.59,35,60015.57,226.54,Streamlined non-volatile analyzer,Lake Nicole,1,Cameroon,2016-03-18 13:22:35,0
41.49,52,32635.7,164.83,Mandatory disintermediate utilization,South John,0,Burundi,2016-05-20 08:49:33,1
87.29,36,61628.72,209.93,Future-proofed methodical protocol,Pamelamouth,1,Korea,2016-03-23 09:43:43,0
41.39,41,68962.32,167.22,Exclusive neutral parallelism,Harperborough,0,Tokelau,2016-06-13 17:27:09,1
78.74,28,64828.0,204.79,Public-key foreground groupware,Port Danielleberg,1,Monaco,2016-05-27 15:25:52,0
48.53,28,38067.08,134.14,Ameliorated client-driven forecast,West Jeremyside,1,Tuvalu,2016-02-08 10:46:14,1
51.95,52,58295.82,129.23,Monitored systematic hierarchy,South Cathyfurt,0,Greece,2016-07-19 08:32:10,1
70.2,34,32708.94,119.2,Open-architected impactful productivity,Palmerside,0,British Virgin Islands,2016-04-14 05:08:35,1
76.02,22,46179.97,209.82,Business-focused value-added definition,West Guybury,0,Bouvet Island (Bouvetoya),2016-01-27 12:38:16,0
67.64,35,51473.28,267.01,Programmable asymmetric data-warehouse,Phelpschester,1,Peru,2016-07-02 20:23:15,0
86.41,28,45593.93,207.48,Digitized static capability,Lake Melindamouth,1,Aruba,2016-03-01 22:13:37,0
59.05,57,25583.29,169.23,Digitized global capability,North Richardburgh,1,Maldives,2016-07-15 05:05:14,1
55.6,23,30227.98,212.58,Multi-layered 4thgeneration knowledge user,Port Cassie,0,Senegal,2016-01-14 14:00:09,1
57.64,57,45580.92,133.81,Synchronized dedicated service-desk,New Thomas,1,Dominica,2016-03-15 03:12:25,1
84.37,30,61389.5,201.58,Synchronized systemic hierarchy,Johnstad,0,Luxembourg,2016-04-12 03:26:39,0
62.26,53,56770.79,125.45,Profound stable product,West Aprilport,1,Montenegro,2016-04-07 15:18:10,1
65.82,39,76435.3,221.94,Reactive demand-driven capacity,Kellytown,0,Ukraine,2016-02-09 05:28:18,0
50.43,46,57425.87,119.32,Persevering needs-based open architecture,Charlesport,1,Saint Helena,2016-05-07 17:11:49,1
38.93,39,27508.41,162.08,Intuitive exuding service-desk,Millerchester,0,Liberia,2016-03-11 06:49:10,1
84.98,29,57691.95,202.61,Innovative user-facing extranet,Mackenziemouth,0,Russian Federation,2016-04-27 09:27:58,0
64.24,30,59784.18,252.36,Front-line intermediate database,Zacharystad,0,Tunisia,2016-04-16 11:53:43,0
82.52,32,66572.39,198.11,Persevering exuding system engine,North Joshua,1,Turkmenistan,2016-05-08 15:38:46,0
81.38,31,64929.61,212.3,Balanced dynamic application,Bowenview,0,Saint Helena,2016-02-08 00:23:38,0
80.47,25,57519.64,204.86,Reduced global support,Jamesberg,0,Niger,2016-02-11 13:26:22,0
37.68,52,53575.48,172.83,Organic leadingedge secured line,Lake Cassandraport,1,Turkmenistan,2016-02-17 13:16:33,1
69.62,20,50983.75,202.25,Business-focused encompassing neural-net,New Sharon,1,Qatar,2016-02-26 22:46:43,0
85.4,43,67058.72,198.72,Triple-buffered demand-driven alliance,Johnport,0,Sri Lanka,2016-06-08 18:54:01,0
44.33,37,52723.34,123.72,Visionary maximized process improvement,Hamiltonfort,1,Trinidad and Tobago,2016-01-08 09:32:26,1
48.01,46,54286.1,119.93,Centralized 24/7 installation,West Christopher,0,Italy,2016-04-25 11:01:54,1
73.18,23,61526.25,196.71,Organized static focus group,Hollandberg,1,British Virgin Islands,2016-04-04 07:07:46,0
79.94,28,58526.04,225.29,Visionary reciprocal circuit,Odomville,0,United Kingdom,2016-05-03 21:19:58,0
33.33,45,53350.11,193.58,Pre-emptive value-added workforce,East Samanthashire,1,Guinea-Bissau,2016-01-17 09:31:36,1
50.33,50,62657.53,133.2,Sharable analyzing alliance,South Lauraton,1,Micronesia,2016-03-02 04:57:51,1
62.31,47,62722.57,119.3,Team-oriented encompassing portal,Amandahaven,0,Turkey,2016-02-14 07:36:58,1
80.6,31,67479.62,177.55,Sharable bottom-line solution,Thomasview,0,Croatia,2016-04-07 03:56:16,0
65.19,36,75254.88,150.61,Cross-group regional website,Garciaside,0,Israel,2016-02-17 11:42:00,1
44.98,49,52336.64,129.31,Organized global model,Port Sarahshire,0,Svalbard & Jan Mayen Islands,2016-04-10 00:13:47,1
77.63,29,56113.37,239.22,Upgradable asynchronous circuit,Port Gregory,0,Azerbaijan,2016-02-14 17:05:15,0
41.82,41,24852.9,156.36,Phased transitional instruction set,Brendachester,0,Iran,2016-05-26 22:49:47,1
85.61,27,47708.42,183.43,Customer-focused empowering ability,Lake Amy,0,Burundi,2016-04-30 08:07:13,0
85.84,34,64654.66,192.93,Front-line heuristic data-warehouse,Lake Annashire,1,Saint Vincent and the Grenadines,2016-06-15 05:30:13,0
72.08,29,71228.44,169.5,Stand-alone national attitude,Smithburgh,0,Burundi,2016-03-09 14:45:33,0
86.06,32,61601.05,178.92,Focused upward-trending core,North Leonmouth,1,Bulgaria,2016-03-31 20:55:22,0
45.96,45,66281.46,141.22,Streamlined cohesive conglomeration,Robertfurt,0,Christmas Island,2016-06-03 00:55:23,1
62.42,29,73910.9,198.5,Upgradable optimizing toolset,Jasminefort,1,Canada,2016-03-10 23:36:03,0
63.89,40,51317.33,105.22,Synchronized user-facing core,Jensenborough,0,Rwanda,2016-01-08 00:17:27,1
35.33,32,51510.18,200.22,Organized client-driven alliance,Bradleyburgh,0,Turks and Caicos Islands,2016-06-05 22:11:34,1
75.74,25,61005.87,215.25,Ergonomic multi-state structure,New Sheila,1,Tunisia,2016-01-16 11:35:01,0
78.53,34,32536.98,131.72,Synergized multimedia emulation,North Regina,0,Norfolk Island,2016-04-22 20:10:22,1
46.13,31,60248.97,139.01,Customer-focused optimizing moderator,Davidmouth,0,Bouvet Island (Bouvetoya),2016-02-01 09:00:55,1
69.01,46,74543.81,222.63,Advanced full-range migration,New Michaeltown,0,Turks and Caicos Islands,2016-07-07 13:37:34,0
55.35,39,75509.61,153.17,De-engineered object-oriented protocol,East Tammie,1,Cook Islands,2016-03-08 00:37:54,1
33.21,43,42650.32,167.07,Polarized clear-thinking budgetary management,Wilcoxport,1,Turkey,2016-05-10 17:39:06,1
38.46,42,58183.04,145.98,Customizable 6thgeneration knowledge user,East Michaelmouth,1,Guatemala,2016-04-06 11:24:21,1
64.1,22,60465.72,215.93,Seamless object-oriented structure,East Tiffanyport,0,Cote d'Ivoire,2016-04-01 16:21:05,0
49.81,35,57009.76,120.06,Seamless real-time array,Ramirezhaven,1,Faroe Islands,2016-01-05 04:18:46,1
82.73,33,54541.56,238.99,Grass-roots impactful system engine,Cranemouth,1,Qatar,2016-05-20 21:31:24,0
56.14,38,32689.04,113.53,Devolved tangible approach,Lake Edward,1,Ireland,2016-02-03 07:59:16,1
55.13,45,55605.92,111.71,Customizable executive software,Lake Conniefurt,0,Ukraine,2016-02-17 21:55:29,1
78.11,27,63296.87,209.25,Progressive analyzing attitude,East Shawnchester,1,Moldova,2016-01-30 16:10:04,0
73.46,28,65653.47,222.75,Innovative executive encoding,West Joseph,1,Nicaragua,2016-05-15 14:41:49,0
56.64,38,61652.53,115.91,Down-sized uniform info-mediaries,Lake Christopherfurt,0,Montserrat,2016-01-05 17:56:52,1
68.94,54,30726.26,138.71,Streamlined next generation implementation,East Tylershire,0,Timor-Leste,2016-04-19 07:34:28,1
70.79,31,74535.94,184.1,Distributed tertiary system engine,Sharpberg,0,Bouvet Island (Bouvetoya),2016-03-15 15:49:14,0
57.76,41,47861.93,105.15,Triple-buffered scalable groupware,Lake Dustin,0,Puerto Rico,2016-06-12 15:25:44,1
77.51,36,73600.28,200.55,Total 5thgeneration encoding,North Kristine,0,Central African Republic,2016-07-01 04:41:57,0
52.7,34,58543.94,118.6,Integrated human-resource encoding,Grahamberg,1,Venezuela,2016-05-08 12:12:04,1
57.7,34,42696.67,109.07,Phased dynamic customer loyalty,New Tina,0,Australia,2016-03-14 23:13:11,1
56.89,37,37334.78,109.29,Open-source coherent policy,Nelsonfurt,1,Wallis and Futuna,2016-05-25 00:19:57,1
69.9,43,71392.53,138.35,Down-sized modular intranet,Christopherport,0,Jersey,2016-05-13 11:51:10,1
55.79,24,59550.05,149.67,Pre-emptive content-based focus group,Port Sarahhaven,0,Puerto Rico,2016-02-20 20:47:05,1
70.03,26,64264.25,227.72,Versatile 4thgeneration system engine,Bradleyborough,1,Samoa,2016-05-22 20:49:37,0
50.08,40,64147.86,125.85,Ergonomic full-range time-frame,Whiteport,1,Greece,2016-04-10 02:02:36,1
43.67,31,25686.34,166.29,Automated directional function,New Theresa,1,Antarctica (the territory South of 60 deg S),2016-02-28 06:41:44,1
72.84,26,52968.22,238.63,Progressive empowering alliance,Wongland,0,Albania,2016-07-08 21:18:32,0
45.72,36,22473.08,154.02,Versatile homogeneous capacity,Williammouth,1,Hong Kong,2016-04-19 15:14:58,1
39.94,41,64927.19,156.3,Function-based optimizing protocol,Williamsborough,0,Lithuania,2016-01-08 22:47:10,1
35.61,46,51868.85,158.22,Up-sized secondary software,North Michael,0,Egypt,2016-03-28 08:46:26,1
79.71,34,69456.83,211.65,Seamless holistic time-frame,Benjaminchester,1,Bangladesh,2016-07-02 14:57:53,0
41.49,53,31947.65,169.18,Persevering reciprocal firmware,Hernandezville,0,Western Sahara,2016-07-03 09:22:30,1
63.6,23,51864.77,235.28,Centralized logistical secured line,Youngburgh,1,Serbia,2016-06-01 09:27:34,0
89.91,40,59593.56,194.23,Innovative background conglomeration,Wallacechester,0,Maldives,2016-07-09 14:55:36,0
68.18,21,48376.14,218.17,Switchable 3rdgeneration hub,Sanchezmouth,1,Czech Republic,2016-02-09 22:04:54,0
66.49,20,56884.74,202.16,Polarized 6thgeneration info-mediaries,Bradshawborough,0,Guernsey,2016-06-10 11:31:33,0
80.49,40,67186.54,229.12,Balanced heuristic approach,Amyhaven,1,Tanzania,2016-02-14 03:50:52,0
72.23,25,46557.92,241.03,Focused 24hour implementation,Marcushaven,1,Bhutan,2016-07-05 17:17:49,0
42.39,42,66541.05,150.99,De-engineered mobile infrastructure,Erinton,0,Christmas Island,2016-04-28 05:50:25,1
47.53,30,33258.09,135.18,Customer-focused upward-trending contingency,Hughesport,0,Guinea,2016-04-03 05:10:31,1
74.02,32,72272.9,210.54,Operative system-worthy protocol,Johnstad,0,Micronesia,2016-03-09 14:57:11,0
66.63,60,60333.38,176.98,User-friendly upward-trending intranet,New Lucasburgh,0,Madagascar,2016-01-16 23:37:51,1
63.24,53,65229.13,235.78,Future-proofed holistic superstructure,Michelleside,1,Lebanon,2016-07-03 04:33:41,1
71.0,22,56067.38,211.87,Extended systemic policy,Andersonton,0,Eritrea,2016-03-14 06:46:14,0
46.13,46,37838.72,123.64,Horizontal hybrid challenge,New Rachel,1,Guyana,2016-01-09 05:44:56,1
69.0,32,72683.35,221.21,Virtual composite model,Port Susan,1,Trinidad and Tobago,2016-02-11 04:37:34,0
76.99,31,56729.78,244.34,Switchable mobile framework,West Angelabury,1,Jersey,2016-06-22 07:33:21,0
72.6,55,66815.54,162.95,Focused intangible moderator,Port Christopherborough,0,United Arab Emirates,2016-07-13 16:12:24,1
61.88,42,60223.52,112.19,Balanced actuating moderator,Phillipsbury,1,Martinique,2016-07-23 11:46:28,1
84.45,50,29727.79,207.18,Customer-focused transitional strategy,Millerside,0,Somalia,2016-07-13 04:10:53,1
88.97,45,49269.98,152.49,Advanced web-enabled standardization,Lake Jessica,0,Bhutan,2016-06-11 18:32:12,1
86.19,31,57669.41,210.26,Pre-emptive executive knowledgebase,Lopezmouth,1,Greece,2016-05-08 12:51:00,0
49.58,26,56791.75,231.94,Self-enabling holistic process improvement,Johnsport,0,Benin,2016-04-07 16:02:02,0
77.65,27,63274.88,212.79,Horizontal client-driven hierarchy,South Ronald,0,Papua New Guinea,2016-02-04 13:30:32,0
37.75,36,35466.8,225.24,Polarized dynamic throughput,South Daniel,0,Uzbekistan,2016-02-26 19:48:23,1
62.33,43,68787.09,127.11,Devolved zero administration intranet,Suzannetown,0,South Africa,2016-06-21 13:15:21,1
79.57,31,61227.59,230.93,User-friendly asymmetric info-mediaries,Lisaberg,0,Egypt,2016-05-17 04:27:31,0
80.31,44,56366.88,127.07,Cross-platform regional task-force,Brianfurt,0,Hungary,2016-04-18 15:54:33,1
89.05,45,57868.44,206.98,Polarized bandwidth-monitored moratorium,Stewartbury,0,Falkland Islands (Malvinas),2016-04-03 10:07:56,0
70.41,27,66618.21,223.03,Centralized systematic knowledgebase,Benjaminchester,0,Dominica,2016-04-04 21:30:46,0
67.36,37,73104.47,233.56,Future-proofed grid-enabled implementation,North Wesleychester,0,Jersey,2016-07-06 16:00:33,0
46.98,50,21644.91,175.37,Down-sized well-modulated archive,East Michelleberg,0,Lithuania,2016-05-04 09:00:24,1
41.67,36,53817.02,132.55,Realigned zero tolerance emulation,Port Eric,0,Saint Martin,2016-06-13 18:50:00,1
51.24,36,76368.31,176.73,Versatile transitional monitoring,Timothyfurt,0,Cuba,2016-01-03 16:01:40,1
75.7,29,67633.44,215.44,Profound zero administration instruction set,Port Jeffrey,0,United States Minor Outlying Islands,2016-01-14 00:23:10,0
43.49,47,50335.46,127.83,User-centric intangible task-force,Guzmanland,0,Belize,2016-01-12 10:07:29,1
49.89,39,17709.98,160.03,Enhanced system-worthy application,East Michele,1,Belize,2016-04-16 12:09:25,1
38.37,36,41229.16,140.46,Multi-layered user-facing paradigm,East John,0,Antarctica (the territory South of 60 deg S),2016-05-13 06:09:28,1
38.52,38,42581.23,137.28,Customer-focused 24/7 concept,Lesliebury,1,Saint Vincent and the Grenadines,2016-03-27 23:59:06,1
71.89,23,61617.98,172.81,Function-based transitional complexity,Patriciahaven,1,Kuwait,2016-02-03 23:47:56,0
75.8,38,70575.6,146.19,Progressive clear-thinking open architecture,Ashleychester,1,Thailand,2016-04-18 11:23:05,0
83.86,31,64122.36,190.25,Up-sized executive moderator,Lake Josetown,0,Gibraltar,2016-02-05 19:06:01,0
37.51,30,52097.32,163.0,Re-contextualized optimal service-desk,Debraburgh,1,Holy See (Vatican City State),2016-03-21 18:46:41,1
55.6,44,65953.76,124.38,Fully-configurable neutral open system,New Debbiestad,1,Korea,2016-06-14 11:59:58,1
83.67,44,60192.72,234.26,Upgradable system-worthy array,West Shaun,1,Saint Helena,2016-02-06 23:08:57,0
69.08,41,77460.07,210.6,Ergonomic client-driven application,Kimberlyhaven,0,Turks and Caicos Islands,2016-03-12 01:39:19,0
37.47,44,45716.48,141.89,Realigned content-based leverage,Port Lawrence,1,Czech Republic,2016-01-26 03:56:18,1
56.04,49,65120.86,128.95,Decentralized real-time circuit,West Ricardo,1,Netherlands,2016-02-07 08:02:31,1
70.92,41,49995.63,108.16,Polarized modular function,Lake Jose,1,Belarus,2016-05-05 07:58:22,1
49.78,46,71718.51,152.24,Enterprise-wide client-driven contingency,Heatherberg,0,Dominica,2016-06-29 02:43:29,1
68.61,57,61770.34,150.29,Diverse modular interface,South George,0,South Africa,2016-04-10 19:48:01,1
58.18,25,69112.84,176.28,Polarized analyzing concept,Tinachester,1,New Zealand,2016-02-10 06:37:56,0
78.54,35,72524.86,172.1,Multi-channeled asynchronous open system,Port Jodi,0,Togo,2016-05-28 20:41:50,0
37.0,48,36782.38,158.22,Function-based context-sensitive secured line,Jonathantown,1,Kenya,2016-03-24 06:36:52,1
65.4,33,66699.12,247.31,Adaptive 24hour Graphic Interface,Sylviaview,0,Palau,2016-02-12 22:51:08,0
79.52,27,64287.78,183.48,Automated coherent flexibility,East Timothyport,1,Timor-Leste,2016-06-10 10:11:00,0
87.98,38,56637.59,222.11,Focused scalable complexity,West Roytown,1,Cambodia,2016-03-31 10:44:46,0
44.64,36,55787.58,127.01,Up-sized incremental encryption,Codyburgh,0,Belize,2016-02-14 06:51:43,1
41.73,28,61142.33,202.18,Sharable dedicated Graphic Interface,Port Erikhaven,1,Cuba,2016-01-07 19:16:05,1
80.46,27,61625.87,207.96,Digitized zero administration paradigm,Port Chasemouth,1,Costa Rica,2016-02-04 02:13:52,0
75.55,36,73234.87,159.24,Managed grid-enabled standardization,Ramirezside,0,Liechtenstein,2016-05-09 02:58:58,1
76.32,35,74166.24,195.31,Networked foreground definition,East Michaeltown,1,Korea,2016-06-23 00:16:02,0
82.68,33,62669.59,222.77,Re-engineered exuding frame,West Courtney,1,Ukraine,2016-06-20 09:35:02,0
72.01,31,57756.89,251.0,Horizontal multi-state interface,West Michaelhaven,0,Angola,2016-02-29 12:31:57,0
75.83,24,58019.64,162.44,Diverse stable circuit,Walshhaven,0,Nauru,2016-01-17 15:10:31,0
41.28,50,50960.08,140.39,Universal 24/7 implementation,East Rachelview,0,Equatorial Guinea,2016-01-29 03:54:19,1
34.66,32,48246.6,194.83,Customer-focused multi-tasking Internet solution,Curtisport,0,Mongolia,2016-07-14 12:07:10,1
66.18,55,28271.84,143.42,Vision-oriented contextually-based extranet,Frankbury,0,Svalbard & Jan Mayen Islands,2016-01-10 23:14:30,1
86.06,31,53767.12,219.72,Extended local methodology,Timothytown,1,Timor-Leste,2016-04-28 18:34:56,0
59.59,42,43662.1,104.78,Re-engineered demand-driven capacity,Samanthaland,1,Brazil,2016-07-06 18:36:01,1
86.69,34,62238.58,198.56,Customer-focused attitude-oriented instruction set,South Jennifer,0,Chad,2016-05-27 06:19:27,0
43.77,52,49030.03,138.55,Synergized hybrid time-frame,Kyleborough,1,Portugal,2016-01-25 07:39:41,1
71.84,47,76003.47,199.79,Advanced exuding conglomeration,North Randy,1,Malawi,2016-05-08 22:47:18,0
80.23,31,68094.85,196.23,Secured clear-thinking middleware,South Daniellefort,0,Qatar,2016-03-19 14:23:45,0
74.41,26,64395.85,163.05,Right-sized value-added initiative,Dianashire,0,Singapore,2016-07-23 04:37:05,0
63.36,48,70053.27,137.43,Centralized tertiary pricing structure,East Eric,0,Guinea,2016-06-23 01:22:43,1
71.74,35,72423.97,227.56,Multi-channeled reciprocal artificial intelligence,Hammondport,0,Kazakhstan,2016-07-19 18:06:22,0
60.72,44,42995.8,105.69,Synergized context-sensitive database,Jacobstad,0,Kuwait,2016-02-28 18:52:44,1
72.04,22,60309.58,199.43,Realigned systematic function,Hernandezfort,0,Rwanda,2016-02-10 06:52:07,0
44.57,31,38349.78,133.17,Adaptive context-sensitive application,Joneston,1,China,2016-03-27 09:11:10,1
85.86,34,63115.34,208.23,Networked high-level structure,New Jeffreychester,0,Bouvet Island (Bouvetoya),2016-05-23 02:15:04,0
39.85,38,31343.39,145.96,Profit-focused dedicated utilization,East Stephen,0,Vietnam,2016-01-03 03:22:15,1
84.53,27,40763.13,168.34,Stand-alone tangible moderator,Turnerchester,0,Guatemala,2016-01-04 21:48:38,1
62.95,60,36752.24,157.04,Polarized tangible collaboration,Youngfort,0,Peru,2016-05-24 13:30:38,1
67.58,41,65044.59,255.61,Focused high-level conglomeration,Ingramberg,1,Mayotte,2016-02-01 19:42:40,0
85.56,29,53673.08,210.46,Advanced modular Local Area Network,South Denisefurt,0,Samoa,2016-06-05 13:16:24,0
46.88,54,43444.86,136.64,Virtual scalable secured line,Port Melissaberg,0,Singapore,2016-02-04 08:53:37,1
46.31,57,44248.52,153.98,Front-line fault-tolerant intranet,Bernardton,1,Jamaica,2016-03-24 13:37:53,1
77.95,31,62572.88,233.65,Inverse asymmetric instruction set,Port Mathew,1,Bahamas,2016-06-02 21:02:22,0
84.73,30,39840.55,153.76,Synchronized leadingedge help-desk,Aliciatown,0,Canada,2016-02-21 07:42:48,1
39.86,36,32593.59,145.85,Total 5thgeneration standardization,Josephstad,0,Algeria,2016-06-26 17:16:26,1
50.08,30,41629.86,123.91,Sharable grid-enabled matrix,West Ericfurt,0,Fiji,2016-01-03 05:34:33,1
60.23,35,43313.73,106.86,Balanced asynchronous hierarchy,New Brendafurt,0,Kenya,2016-03-08 18:00:43,1
60.7,49,42993.48,110.57,Monitored object-oriented Graphic Interface,Port Julie,1,Argentina,2016-06-19 03:19:44,1
43.67,53,46004.31,143.79,Cloned analyzing artificial intelligence,South Tiffanyton,1,Bouvet Island (Bouvetoya),2016-07-21 21:16:35,1
77.2,33,49325.48,254.05,Persistent homogeneous framework,North Elizabeth,1,Philippines,2016-02-12 20:36:40,0
71.86,32,51633.34,116.53,Face-to-face even-keeled website,Kentmouth,0,Senegal,2016-05-17 06:14:20,1
44.78,45,63363.04,137.24,Extended context-sensitive monitoring,West Casey,1,Suriname,2016-07-09 11:04:54,1
78.57,36,64045.93,239.32,Exclusive client-driven model,East Henry,1,Liberia,2016-03-27 02:35:29,0
73.41,31,73049.3,201.26,Profound executive flexibility,Hollyfurt,1,Guam,2016-01-16 08:01:40,0
77.05,27,66624.6,191.14,Reduced bi-directional strategy,North Anna,0,United Arab Emirates,2016-01-21 23:48:29,0
66.4,40,77567.85,214.42,Digitized heuristic solution,Port Destiny,0,Antigua and Barbuda,2016-06-05 00:29:13,0
69.35,29,53431.35,252.77,Seamless 4thgeneration contingency,Ianmouth,1,Argentina,2016-02-13 15:37:36,0
35.65,40,31265.75,172.58,Seamless intangible secured line,North Johntown,1,Georgia,2016-05-10 07:22:37,1
70.04,31,74780.74,183.85,Intuitive radical forecast,Hannahside,1,Jordan,2016-03-27 03:59:26,0
69.78,29,70410.11,218.79,Multi-layered non-volatile Graphical User Interface,Wilsonburgh,0,Saudi Arabia,2016-05-24 18:35:58,0
58.22,29,37345.24,120.9,User-friendly client-server instruction set,North Russellborough,0,South Africa,2016-02-11 02:40:02,1
76.9,28,66107.84,212.67,Synchronized multimedia model,Murphymouth,0,Croatia,2016-04-22 08:31:24,0
84.08,30,62336.39,187.36,Face-to-face intermediate approach,Carterburgh,1,Fiji,2016-01-13 02:58:27,0
59.51,58,39132.64,140.83,Assimilated fault-tolerant hub,Penatown,0,Australia,2016-06-16 02:01:24,1
40.15,38,38745.29,134.88,Exclusive disintermediate task-force,Joechester,1,Sao Tome and Principe,2016-06-27 18:37:04,1
76.81,28,65172.22,217.85,Managed zero tolerance concept,East Paul,1,Fiji,2016-07-03 12:57:03,0
41.89,38,68519.96,163.38,Compatible systemic function,Hartmanchester,0,Cyprus,2016-02-03 04:21:14,1
76.87,27,54774.77,235.35,Configurable fault-tolerant monitoring,Mcdonaldfort,1,Kyrgyz Republic,2016-05-29 21:17:10,0
67.28,43,76246.96,155.8,Future-proofed coherent hardware,North Mercedes,1,Pakistan,2016-04-03 21:13:46,1
81.98,40,65461.92,229.22,Ameliorated upward-trending definition,Taylorberg,0,Seychelles,2016-04-15 11:51:14,0
66.01,23,34127.21,151.95,Front-line tangible alliance,Hansenmouth,0,Samoa,2016-06-21 03:14:41,1
61.57,53,35253.98,125.94,Progressive 24hour forecast,Bradyfurt,1,Bulgaria,2016-03-14 14:13:05,1
53.3,34,44893.71,111.94,Self-enabling optimal initiative,West Jessicahaven,0,Mauritania,2016-05-06 21:07:31,1
34.87,40,59621.02,200.23,Configurable logistical Graphical User Interface,Davilachester,0,Czech Republic,2016-06-12 17:52:43,1
43.6,38,20856.54,170.49,Virtual bandwidth-monitored initiative,North Ricardotown,0,Chile,2016-01-11 07:36:22,1
77.88,37,55353.41,254.57,Multi-tiered human-resource structure,Melissafurt,0,Poland,2016-07-02 00:24:22,0
75.83,27,67516.07,200.59,Managed upward-trending instruction set,East Brianberg,0,Estonia,2016-03-04 10:13:48,0
49.95,39,68737.75,136.59,Cloned object-oriented benchmark,Millerbury,0,Turkmenistan,2016-03-24 09:12:52,1
60.94,41,76893.84,154.97,Fundamental fault-tolerant neural-net,Garciaview,0,Latvia,2016-02-14 07:30:24,1
89.15,42,59886.58,171.07,Phased zero administration success,Townsendfurt,0,Fiji,2016-04-25 07:30:21,0
78.7,30,53441.69,133.99,Compatible intangible customer loyalty,Williamstad,0,Turkey,2016-02-10 19:20:51,1
57.35,29,41356.31,119.84,Distributed 3rdgeneration definition,West Connor,0,Kazakhstan,2016-04-23 14:34:38,1
34.86,38,49942.66,154.75,Pre-emptive cohesive budgetary management,West Justin,0,Bahrain,2016-06-18 17:56:32,1
70.68,31,74430.08,199.08,Configurable multi-state utilization,Robertbury,0,Colombia,2016-07-17 01:58:53,0
76.06,23,58633.63,201.04,Diverse multi-tasking parallelism,New Tinamouth,0,Brunei Darussalam,2016-04-27 04:28:17,0
66.67,33,72707.87,228.03,Horizontal content-based synergy,Turnerview,1,Taiwan,2016-04-21 20:29:35,0
46.77,32,31092.93,136.4,Multi-tiered maximized archive,Reneechester,1,Serbia,2016-03-23 06:00:15,1
62.42,38,74445.18,143.94,Diverse executive groupware,West Tinashire,0,Saint Pierre and Miquelon,2016-07-19 07:59:18,1
78.32,28,49309.14,239.52,Synergized cohesive array,Jamesfurt,0,Australia,2016-06-26 11:52:18,1
37.32,50,56735.14,199.25,Versatile dedicated software,New Nancy,1,Chad,2016-03-30 23:40:52,1
40.42,45,40183.75,133.9,Stand-alone reciprocal synergy,Lisamouth,1,Norway,2016-03-16 07:59:37,1
76.77,36,58348.41,123.51,Universal even-keeled analyzer,Harveyport,0,Turks and Caicos Islands,2016-05-04 00:01:33,1
65.65,30,72209.99,158.05,Up-sized tertiary contingency,Ramosstad,0,Finland,2016-07-02 21:22:23,0
74.32,33,62060.11,128.17,Monitored real-time superstructure,North Kevinside,0,South Africa,2016-05-23 21:14:38,1
73.27,32,67113.46,234.75,Streamlined analyzing initiative,Haleview,1,Martinique,2016-01-29 20:16:54,0
80.03,44,24030.06,150.84,Automated static concept,Christinetown,0,Afghanistan,2016-07-23 14:47:23,1
53.68,47,56180.93,115.26,Operative stable moderator,New Michael,1,Micronesia,2016-02-16 09:11:27,1
85.84,32,62204.93,192.85,Up-sized 6thgeneration moratorium,Jonesland,1,French Southern Territories,2016-06-09 21:43:05,0
85.03,30,60372.64,204.52,Expanded clear-thinking core,North Shannon,0,Philippines,2016-06-19 09:24:35,0
70.44,24,65280.16,178.75,Polarized attitude-oriented superstructure,New Sonialand,1,Algeria,2016-06-06 21:26:51,0
81.22,53,34309.24,223.09,Networked coherent interface,Port Jason,1,San Marino,2016-01-07 13:25:21,0
39.96,45,59610.81,146.13,Enhanced homogeneous moderator,East Barbara,1,Guernsey,2016-04-15 06:08:35,1
57.05,41,50278.89,269.96,Seamless full-range website,Port Erinberg,1,Sierra Leone,2016-01-09 03:45:19,1
42.44,56,43450.11,168.27,Profit-focused attitude-oriented task-force,Petersonfurt,0,Tajikistan,2016-02-10 15:23:17,1
62.2,25,25408.21,161.16,Cross-platform multimedia algorithm,New Lindaberg,0,Liechtenstein,2016-04-24 13:42:15,1
76.7,36,71136.49,222.25,Open-source coherent monitoring,West Russell,0,Ecuador,2016-06-12 05:31:19,0
61.22,45,63883.81,119.03,Streamlined logistical secured line,South Adam,1,Switzerland,2016-01-05 09:42:22,1
84.54,33,64902.47,204.02,Synchronized stable complexity,North Tracyport,1,Moldova,2016-03-02 10:07:43,0
46.08,30,66784.81,164.63,Synergistic value-added extranet,Brownport,1,Finland,2016-07-21 10:54:35,1
56.7,48,62784.85,123.13,Progressive non-volatile neural-net,Port Crystal,0,France,2016-01-09 04:53:22,1
81.03,28,63727.5,201.15,Persevering tertiary capability,Masonhaven,0,Venezuela,2016-01-06 13:20:01,0
80.91,32,61608.23,231.42,Enterprise-wide bi-directional secured line,Derrickhaven,0,Cuba,2016-01-31 04:10:20,0
40.06,38,56782.18,138.68,Organized contextually-based customer loyalty,Olsonstad,1,Peru,2016-06-11 08:38:16,1
83.47,39,64447.77,226.11,Total directional approach,New Brandy,0,Turkey,2016-05-15 20:48:40,0
73.84,31,42042.95,121.05,Programmable uniform productivity,South Jasminebury,0,Albania,2016-06-18 17:23:26,1
74.65,28,67669.06,212.56,Robust transitional ability,East Timothy,0,French Southern Territories,2016-03-17 05:00:12,0
60.25,35,54875.95,109.77,De-engineered fault-tolerant database,Charlottefort,0,Papua New Guinea,2016-06-29 13:35:05,1
59.21,35,73347.67,144.62,Managed disintermediate matrices,Lake Beckyburgh,1,Liechtenstein,2016-02-02 08:55:26,1
43.02,44,50199.77,125.22,Configurable bottom-line application,West Lindseybury,0,Thailand,2016-04-13 05:42:52,1
84.04,38,50723.67,244.55,Self-enabling didactic pricing structure,West Alyssa,0,Malaysia,2016-07-20 09:27:24,0
70.66,43,63450.96,120.95,Versatile scalable encryption,Lake Craigview,1,Mauritius,2016-02-26 04:57:14,1
70.58,26,56694.12,136.94,Proactive next generation knowledge user,Lake David,0,Algeria,2016-02-26 09:18:48,1
72.44,34,70547.16,230.14,Customizable tangible hierarchy,Bruceburgh,0,Christmas Island,2016-04-15 14:45:48,0
40.17,26,47391.95,171.31,Visionary asymmetric encryption,South Lauratown,1,Japan,2016-02-01 14:37:34,1
79.15,26,62312.23,203.23,Intuitive explicit conglomeration,Port Robin,0,Greenland,2016-01-20 19:09:37,0
44.49,53,63100.13,168.0,Business-focused real-time toolset,Jacksonburgh,1,Sao Tome and Principe,2016-04-23 06:28:43,1
73.04,37,73687.5,221.79,Organic contextually-based focus group,Erinmouth,1,Senegal,2016-06-19 22:26:16,0
76.28,33,52686.47,254.34,Right-sized asynchronous website,Port Aliciabury,0,Guadeloupe,2016-02-15 07:55:10,0
68.88,37,78119.5,179.58,Advanced 5thgeneration capability,Port Whitneyhaven,0,Belgium,2016-02-09 19:37:52,0
73.1,28,57014.84,242.37,Universal asymmetric archive,Jeffreyshire,0,Israel,2016-01-25 07:52:53,0
47.66,29,27086.4,156.54,Devolved responsive structure,Tinaton,0,Honduras,2016-07-18 11:33:31,1
87.3,35,58337.18,216.87,Triple-buffered regional toolset,North Loriburgh,0,Estonia,2016-01-09 07:28:16,0
89.34,32,50216.01,177.78,Object-based executive productivity,Wendyton,1,Paraguay,2016-03-21 21:15:54,0
81.37,26,53049.44,156.48,Business-focused responsive website,Lake Jacqueline,1,Kyrgyz Republic,2016-02-15 12:25:28,0
81.67,28,62927.96,196.76,Visionary analyzing structure,North Christopher,1,Mauritania,2016-03-04 08:48:29,0
46.37,52,32847.53,144.27,De-engineered solution-oriented open architecture,Alexanderfurt,0,French Guiana,2016-01-05 00:02:53,1
54.88,24,32006.82,148.61,Customizable modular Internet solution,West Pamela,0,Northern Mariana Islands,2016-05-15 01:03:06,1
40.67,35,48913.07,133.18,Stand-alone encompassing throughput,West Amanda,0,Lebanon,2016-05-05 09:28:36,1
71.76,35,69285.69,237.39,Customizable zero-defect matrix,South Tomside,0,Saint Pierre and Miquelon,2016-05-26 13:18:30,0
47.51,51,53700.57,130.41,Managed well-modulated collaboration,Bethburgh,1,American Samoa,2016-05-21 01:36:16,1
75.15,22,52011.0,212.87,Universal global intranet,Jamiefort,1,Austria,2016-05-04 12:06:18,0
56.01,26,46339.25,127.26,Re-engineered real-time success,Garciamouth,0,Tonga,2016-07-05 18:59:45,1
82.87,37,67938.77,213.36,Front-line fresh-thinking open system,West Brenda,0,Tonga,2016-06-28 20:13:41,0
45.05,42,66348.95,141.36,Digitized contextually-based product,South Kyle,0,French Southern Territories,2016-05-05 11:09:29,1
60.53,24,66873.9,167.22,Organic interactive support,Combsstad,0,Serbia,2016-03-25 15:17:39,1
50.52,31,72270.88,171.62,Function-based stable alliance,Lake Allenville,0,New Caledonia,2016-01-23 15:02:13,1
84.71,32,61610.05,210.23,Reactive responsive emulation,Greenechester,0,Taiwan,2016-05-29 07:29:27,0
55.2,39,76560.59,159.46,Exclusive zero tolerance alliance,Jordantown,1,United States of America,2016-05-30 07:36:31,1
81.61,33,62667.51,228.76,Enterprise-wide local matrices,Gravesport,0,Morocco,2016-04-17 15:46:03,0
71.55,36,75687.46,163.99,Inverse next generation moratorium,South Troy,1,Suriname,2016-07-20 23:08:28,0
82.4,36,66744.65,218.97,Implemented bifurcated workforce,Lake Patrick,1,Macedonia,2016-06-29 03:07:51,0
73.95,35,67714.82,238.58,Persevering even-keeled help-desk,Millerland,0,Wallis and Futuna,2016-04-10 14:48:35,0
72.07,31,69710.51,226.45,Grass-roots eco-centric instruction set,Port Jessicamouth,0,Chile,2016-04-16 16:38:35,0
80.39,31,66269.49,214.74,Fully-configurable incremental Graphical User Interface,Paulport,0,Gabon,2016-05-03 08:21:23,0
65.8,25,60843.32,231.49,Expanded radical software,Clineshire,1,Gabon,2016-03-18 16:04:59,0
69.97,28,55041.6,250.0,Mandatory 3rdgeneration moderator,Cynthiaside,0,Holy See (Vatican City State),2016-05-22 00:01:58,0
52.62,50,73863.25,176.52,Enterprise-wide foreground emulation,Port Juan,0,Seychelles,2016-02-01 20:30:35,1
39.25,39,62378.05,152.36,Customer-focused incremental system engine,Michellefort,0,Mayotte,2016-01-23 17:39:06,1
77.56,38,63336.85,130.83,Right-sized multi-tasking solution,Port Angelamouth,1,Uganda,2016-05-19 03:52:24,1
33.52,43,42191.61,165.56,Vision-oriented optimizing middleware,Jessicahaven,0,Cambodia,2016-05-09 21:54:38,1
79.81,24,56194.56,178.85,Proactive context-sensitive project,North Daniel,1,Antigua and Barbuda,2016-05-31 11:44:45,1
84.79,33,61771.9,214.53,Managed eco-centric encoding,New Juan,0,Cameroon,2016-03-30 19:09:50,0
82.7,35,61383.79,231.07,Visionary multi-tasking alliance,Amyfurt,0,Somalia,2016-01-09 15:49:28,0
84.88,32,63924.82,186.48,Ameliorated tangible hierarchy,Harrishaven,0,Lebanon,2016-04-18 03:41:56,0
54.92,54,23975.35,161.16,Extended interactive model,Roberttown,0,Saint Pierre and Miquelon,2016-06-13 13:59:51,1
76.56,34,70179.11,221.53,Universal bi-directional extranet,Jeremyshire,1,Dominica,2016-04-23 08:15:31,0
69.74,49,66524.8,243.37,Enhanced maximized access,Birdshire,0,Hungary,2016-03-27 16:41:29,0
75.55,22,41851.38,169.4,Upgradable even-keeled challenge,New Amanda,0,Taiwan,2016-02-19 07:29:30,1
72.19,33,61275.18,250.35,Synchronized national infrastructure,Curtisview,1,Saint Lucia,2016-05-19 11:16:59,0
84.29,41,60638.38,232.54,Re-contextualized systemic time-frame,Jacksonmouth,0,Niue,2016-01-27 20:47:57,0
73.89,39,47160.53,110.68,Horizontal national architecture,North April,0,France,2016-04-20 00:41:53,1
75.84,21,48537.18,186.98,Reactive bi-directional workforce,Hayesmouth,0,Cyprus,2016-02-07 07:41:06,0
73.38,25,53058.91,236.19,Horizontal transitional challenge,South Corey,1,French Southern Territories,2016-04-21 09:30:35,0
80.72,31,68614.98,186.37,Re-engineered neutral success,Juliaport,0,Costa Rica,2016-04-19 05:15:28,0
62.06,44,44174.25,105.0,Adaptive contextually-based methodology,Port Paultown,0,Austria,2016-04-12 14:01:08,1
51.5,34,67050.16,135.31,Configurable dynamic adapter,East Vincentstad,0,Zambia,2016-03-15 11:25:48,1
90.97,37,54520.14,180.77,Multi-lateral empowering throughput,Kimberlytown,0,Congo,2016-02-16 18:21:36,0
86.78,30,54952.42,170.13,Fundamental zero tolerance solution,New Steve,1,United States of America,2016-02-18 23:08:59,0
66.18,35,69476.42,243.61,Proactive asymmetric definition,New Johnberg,0,Pitcairn Islands,2016-03-25 08:40:15,0
84.33,41,54989.93,240.95,Pre-emptive zero tolerance Local Area Network,Shawstad,0,Belize,2016-03-16 00:28:10,0
36.87,36,29398.61,195.91,Self-enabling incremental collaboration,New Rebecca,0,Anguilla,2016-01-28 11:50:40,1
34.78,48,42861.42,208.21,Exclusive even-keeled moratorium,Jeffreyburgh,1,South Africa,2016-03-24 02:01:55,1
76.84,32,65883.39,231.59,Reduced incremental productivity,Faithview,0,Singapore,2016-03-03 22:31:16,0
67.05,25,65421.39,220.92,Realigned scalable standardization,Richardsontown,0,Finland,2016-02-26 09:54:33,0
41.47,31,60953.93,219.79,Secured scalable Graphical User Interface,Port Brookeland,0,Martinique,2016-07-06 15:56:39,1
80.71,26,58476.57,200.58,Team-oriented context-sensitive installation,East Christopherbury,0,Cameroon,2016-06-24 05:50:22,0
80.09,31,66636.84,214.08,Pre-emptive systematic budgetary management,Port Christinemouth,0,Sweden,2016-05-23 21:00:45,0
56.3,49,67430.96,135.24,Fully-configurable high-level implementation,South Meghan,1,New Caledonia,2016-02-03 19:12:51,1
79.36,34,57260.41,245.78,Profound maximized workforce,Hessstad,1,Bosnia and Herzegovina,2016-04-28 22:54:37,0
86.38,40,66359.32,188.27,Cross-platform 4thgeneration focus group,Rhondaborough,1,Singapore,2016-03-19 14:57:00,0
38.94,41,57587.0,142.67,Optional mission-critical functionalities,Lewismouth,1,Falkland Islands (Malvinas),2016-07-15 09:08:42,1
87.26,35,63060.55,184.03,Multi-layered tangible portal,New Paul,0,Bosnia and Herzegovina,2016-05-12 04:35:59,0
75.32,28,59998.5,233.6,Reduced mobile structure,Lake Angela,1,Mauritius,2016-01-01 21:58:55,0
74.38,40,74024.61,220.05,Enhanced zero tolerance Graphic Interface,East Graceland,1,Indonesia,2016-03-13 13:50:25,0
65.9,22,60550.66,211.39,De-engineered tertiary secured line,Hartport,0,Czech Republic,2016-07-16 14:13:54,0
36.31,47,57983.3,168.92,Reverse-engineered well-modulated capability,East Yvonnechester,0,Eritrea,2016-04-18 00:49:33,1
72.23,48,52736.33,115.35,Integrated coherent pricing structure,Burgessside,0,Mexico,2016-07-17 01:13:56,1
88.12,38,46653.75,230.91,Realigned next generation projection,Hurleyborough,0,Gibraltar,2016-02-17 07:05:57,0
83.97,28,56986.73,205.5,Reactive needs-based instruction set,Garychester,1,Haiti,2016-06-16 02:33:22,0
61.09,26,55336.18,131.68,User-friendly well-modulated leverage,East Kevinbury,1,Falkland Islands (Malvinas),2016-04-09 16:31:15,1
65.77,21,42162.9,218.61,Function-based fault-tolerant model,Contrerasshire,1,Eritrea,2016-03-18 17:35:40,0
81.58,25,39699.13,199.39,Decentralized needs-based analyzer,Erikville,0,Hong Kong,2016-05-11 22:02:17,0
37.87,52,56394.82,188.56,Phased analyzing emulation,Robertsonburgh,1,Gambia,2016-05-25 20:10:02,1
76.2,37,75044.35,178.51,Multi-layered fresh-thinking process improvement,Karenton,0,Barbados,2016-02-29 19:26:35,0
60.91,19,53309.61,184.94,Upgradable directional system engine,Port Kathleenfort,0,Nauru,2016-06-09 14:24:06,1
74.49,28,58996.12,237.34,Persevering eco-centric flexibility,Lake Adrian,0,Peru,2016-01-30 16:15:29,0
73.71,23,56605.12,211.38,Inverse local hub,New Sheila,1,El Salvador,2016-02-15 05:35:54,0
78.19,30,62475.99,228.81,Triple-buffered needs-based Local Area Network,Mollyport,0,Libyan Arab Jamahiriya,2016-01-31 06:14:10,0
79.54,44,70492.6,217.68,Centralized multi-state hierarchy,Sandraland,1,Cambodia,2016-01-05 16:34:31,0
74.87,52,43698.53,126.97,Public-key non-volatile implementation,Charlenetown,0,Saint Barthelemy,2016-05-31 02:17:18,1
87.09,36,57737.51,221.98,Synergized coherent interface,Luischester,1,Reunion,2016-04-21 16:10:50,0
37.45,47,31281.01,167.86,Horizontal high-level concept,South Johnnymouth,0,Antigua and Barbuda,2016-04-10 03:30:16,1
49.84,39,45800.48,111.59,Reduced multimedia project,Hannaport,0,Samoa,2016-02-09 07:21:25,1
51.38,59,42362.49,158.56,Object-based modular functionalities,East Anthony,0,Afghanistan,2016-06-17 17:11:16,1
83.4,34,66691.23,207.87,Polarized multimedia system engine,West Daleborough,0,Azerbaijan,2016-05-22 21:54:23,0
38.91,33,56369.74,150.8,Versatile reciprocal structure,Morrismouth,1,Philippines,2016-07-13 07:41:42,1
62.14,41,59397.89,110.93,Upgradable multi-tasking initiative,North Andrewstad,1,Angola,2016-01-23 18:59:21,1
79.72,28,66025.11,193.8,Configurable tertiary budgetary management,Wrightburgh,1,Albania,2016-05-20 12:17:59,0
73.3,36,68211.35,135.72,Adaptive asynchronous attitude,West Tanya,1,Hungary,2016-01-30 04:38:41,1
69.11,42,73608.99,231.48,Face-to-face mission-critical definition,Novaktown,1,Faroe Islands,2016-04-21 12:34:28,0
71.9,54,61228.96,140.15,Inverse zero tolerance customer loyalty,Timothymouth,1,Czech Republic,2016-04-22 20:32:17,1
72.45,29,72325.91,195.36,Centralized 24hour synergy,Robertmouth,1,Svalbard & Jan Mayen Islands,2016-01-11 06:02:27,0
77.07,40,44559.43,261.02,Face-to-face analyzing encryption,Stephenborough,0,Afghanistan,2016-03-01 10:01:35,0
74.62,36,73207.15,217.79,Self-enabling even-keeled methodology,Lake Kurtmouth,0,Rwanda,2016-04-04 08:19:54,0
82.07,25,46722.07,205.38,Function-based optimizing extranet,Lauraburgh,1,Panama,2016-06-20 06:30:06,0
58.6,50,45400.5,113.7,Organic asynchronous hierarchy,Rogerburgh,0,Samoa,2016-01-28 07:10:29,1
36.08,45,41417.27,151.47,Automated client-driven orchestration,Davidside,1,United States Minor Outlying Islands,2016-07-03 04:11:40,1
79.44,26,60845.55,206.79,Public-key zero-defect analyzer,West Thomas,0,Greece,2016-05-15 13:18:34,0
41.73,47,60812.77,144.71,Proactive client-server productivity,Andersonchester,0,Cote d'Ivoire,2016-04-08 22:48:25,1
73.19,25,64267.88,203.74,Cloned incremental matrices,North Ronaldshire,1,Pakistan,2016-01-19 12:18:13,0
77.6,24,58151.87,197.33,Open-architected system-worthy task-force,Greghaven,1,Anguilla,2016-05-26 15:40:26,0
89.0,37,52079.18,222.26,Devolved regional moderator,Jordanmouth,1,Cyprus,2016-01-26 15:56:55,0
69.2,42,26023.99,123.8,Balanced value-added database,Meyersstad,0,Peru,2016-06-17 09:58:46,1
67.56,31,62318.38,125.45,Seamless composite budgetary management,Michelleside,0,Kenya,2016-04-25 21:15:39,1
81.11,39,56216.57,248.19,Total cohesive moratorium,South Robert,1,Chad,2016-07-13 11:41:29,0
80.22,30,61806.31,224.58,Integrated motivating neural-net,New Tyler,0,Kyrgyz Republic,2016-07-05 15:14:10,0
43.63,41,51662.24,123.25,Exclusive zero tolerance frame,Jordanshire,1,Albania,2016-03-15 14:06:17,1
77.66,29,67080.94,168.15,Operative scalable emulation,Reyesland,0,Gabon,2016-06-19 22:08:15,0
74.63,26,51975.41,235.99,Enhanced asymmetric installation,New Traceystad,1,Dominican Republic,2016-07-05 20:16:13,0
49.67,27,28019.09,153.69,Face-to-face reciprocal methodology,Port Brian,0,Zimbabwe,2016-05-09 08:44:55,1
80.59,37,67744.56,224.23,Robust responsive collaboration,Lake Courtney,0,Croatia,2016-07-21 23:14:35,0
83.49,33,66574.0,190.75,Polarized logistical hub,Samuelborough,1,Cambodia,2016-06-03 17:32:47,0
44.46,42,30487.48,132.66,Intuitive zero-defect framework,Christinehaven,1,Mongolia,2016-01-15 19:40:47,1
68.1,40,74903.41,227.73,Reactive composite project,Thomasstad,1,Honduras,2016-02-05 16:50:58,0
63.88,38,19991.72,136.85,Upgradable even-keeled hardware,Kristintown,0,Madagascar,2016-02-29 23:56:06,1
78.83,36,66050.63,234.64,Future-proofed responsive matrix,New Wanda,1,Qatar,2016-05-08 12:08:26,0
79.97,44,70449.04,216.0,Programmable empowering middleware,Mariebury,0,China,2016-07-13 01:48:46,0
80.51,28,64008.55,200.28,Robust dedicated system engine,Christopherville,1,Bangladesh,2016-01-08 02:34:06,0
62.26,26,70203.74,202.77,Public-key mission-critical core,New Jasmine,0,Swaziland,2016-06-08 12:25:49,0
66.99,47,27262.51,124.44,Operative actuating installation,Lopezberg,1,Tanzania,2016-06-15 11:56:41,1
71.05,20,49544.41,204.22,Self-enabling asynchronous knowledge user,Jenniferstad,1,Eritrea,2016-06-13 22:41:45,0
42.05,51,28357.27,174.55,Configurable 24/7 hub,West Eduardotown,1,Canada,2016-06-20 14:20:52,1
50.52,28,66929.03,219.69,Versatile responsive knowledge user,Davisfurt,0,Saint Kitts and Nevis,2016-04-03 06:17:22,1
76.24,40,75524.78,198.32,Managed impactful definition,Bakerhaven,1,Burkina Faso,2016-05-31 23:42:26,0
77.29,27,66265.34,201.24,Grass-roots 4thgeneration forecast,Paulshire,1,Tuvalu,2016-02-15 03:43:55,0
35.98,47,55993.68,165.52,Focused 3rdgeneration pricing structure,West Jane,1,El Salvador,2016-03-10 23:26:54,1
84.95,34,56379.3,230.36,Mandatory dedicated data-warehouse,Lake Brian,0,Madagascar,2016-02-26 17:01:01,0
39.34,43,31215.88,148.93,Proactive radical support,Alvaradoport,0,Bangladesh,2016-04-17 21:39:11,1
87.23,29,51015.11,202.12,Re-engineered responsive definition,Lake Kevin,0,American Samoa,2016-03-26 19:54:16,0
57.24,52,46473.14,117.35,Profound optimizing utilization,Richardsonland,1,Latvia,2016-06-29 21:39:42,1
81.58,41,55479.62,248.16,Cloned explicit middleware,East Sheriville,0,Moldova,2016-01-27 17:55:44,0
56.34,50,68713.7,139.02,Multi-channeled mission-critical success,Port Michealburgh,1,Anguilla,2016-03-17 23:39:28,1
48.73,27,34191.23,142.04,Versatile content-based protocol,Monicaview,0,Bangladesh,2016-07-09 16:23:33,1
51.68,49,51067.54,258.62,Seamless cohesive conglomeration,Katieport,0,Faroe Islands,2016-06-28 12:51:02,1
35.34,45,46693.76,152.86,De-engineered actuating hierarchy,East Brittanyville,0,Taiwan,2016-06-18 16:32:58,1
48.09,33,19345.36,180.42,Balanced motivating help-desk,West Travismouth,0,Heard Island and McDonald Islands,2016-05-28 12:38:37,1
78.68,29,66225.72,208.05,Inverse high-level capability,Leonchester,0,Israel,2016-01-16 16:40:30,0
68.82,20,38609.2,205.64,Cross-platform client-server hierarchy,Ramirezland,1,Bolivia,2016-07-11 15:45:23,0
56.99,40,37713.23,108.15,Sharable optimal capacity,Brownton,0,Bahamas,2016-07-16 23:08:54,1
86.63,39,63764.28,209.64,Face-to-face multimedia success,New Jessicaport,1,Costa Rica,2016-04-06 21:20:07,0
41.18,43,41866.55,129.25,Enterprise-wide incremental Internet solution,New Denisebury,1,Myanmar,2016-07-05 00:54:11,1
71.03,32,57846.68,120.85,Advanced systemic productivity,Keithtown,0,Netherlands Antilles,2016-02-17 23:47:00,1
72.92,29,69428.73,217.1,Customizable mission-critical adapter,Port Melissastad,1,Czech Republic,2016-03-15 17:33:15,0
77.14,24,60283.98,184.88,Horizontal heuristic synergy,Janiceview,1,Iceland,2016-01-21 18:51:01,0
60.7,43,79332.33,192.6,Multi-tiered multi-state moderator,Mataberg,1,Palau,2016-06-06 22:41:24,0
34.3,41,53167.68,160.74,Re-contextualized reciprocal interface,West Melaniefurt,1,Libyan Arab Jamahiriya,2016-05-16 14:50:22,1
83.71,45,64564.07,220.48,Organized demand-driven knowledgebase,Millerfort,1,Kazakhstan,2016-04-17 19:10:56,0
53.38,35,60803.37,120.06,Total local synergy,Alexanderview,1,French Guiana,2016-03-30 01:05:34,1
58.03,31,28387.42,129.33,User-friendly bandwidth-monitored attitude,South Jade,0,Tuvalu,2016-06-29 09:04:31,1
43.59,36,58849.77,132.31,Re-engineered context-sensitive knowledge user,Lake Susan,1,Congo,2016-05-26 13:43:05,1
60.07,42,65963.37,120.75,Total user-facing hierarchy,South Vincentchester,1,United Kingdom,2016-04-15 10:16:49,1
54.43,37,75180.2,154.74,Balanced contextually-based pricing structure,Williamsmouth,1,Luxembourg,2016-05-31 09:06:29,1
81.99,33,61270.14,230.9,Inverse bi-directional knowledge user,Taylorport,0,French Polynesia,2016-02-15 14:13:47,0
60.53,29,56759.48,123.28,Networked even-keeled workforce,Williamsport,0,Papua New Guinea,2016-05-09 10:21:48,1
84.69,31,46160.63,231.85,Right-sized transitional parallelism,Emilyfurt,1,Maldives,2016-07-07 23:32:38,0
88.72,32,43870.51,211.87,Customer-focused system-worthy superstructure,East John,1,Zambia,2016-01-03 17:10:05,0
88.89,35,50439.49,218.8,Balanced 4thgeneration success,East Deborahhaven,1,Cook Islands,2016-07-17 18:55:38,0
69.58,43,28028.74,255.07,Cross-group value-added success,Port Katelynview,0,Congo,2016-04-04 18:36:59,1
85.23,36,64238.71,212.92,Visionary client-driven installation,Paulhaven,1,Senegal,2016-02-27 12:34:19,0
83.55,39,65816.38,221.18,Switchable well-modulated infrastructure,Elizabethmouth,1,Myanmar,2016-06-08 20:13:27,0
56.66,42,72684.44,139.42,Upgradable asymmetric emulation,Lake Jesus,0,Dominican Republic,2016-02-20 10:52:51,1
56.39,27,38817.4,248.12,Configurable tertiary capability,North Tylerland,1,Bahrain,2016-03-23 21:06:51,0
76.24,27,63976.44,214.42,Monitored dynamic instruction set,Munozberg,0,Puerto Rico,2016-06-07 01:29:06,0
57.64,36,37212.54,110.25,Robust web-enabled attitude,North Maryland,1,Chile,2016-01-18 15:18:01,1
78.18,23,52691.79,167.67,Customer-focused full-range neural-net,West Barbara,0,Bolivia,2016-06-09 19:32:27,0
46.04,32,65499.93,147.92,Universal transitional Graphical User Interface,Andrewborough,0,Serbia,2016-05-30 20:07:59,1
79.4,35,63966.72,236.87,User-centric intangible contingency,New Gabriel,0,Malaysia,2016-04-01 09:21:14,0
36.44,39,52400.88,147.64,Configurable disintermediate throughput,Port Patrickton,1,Estonia,2016-05-31 06:21:02,1
53.14,38,49111.47,109.0,Automated web-enabled migration,West Julia,1,Greenland,2016-07-03 22:13:19,1
32.84,40,41232.89,171.72,Triple-buffered 3rdgeneration migration,New Keithburgh,0,Trinidad and Tobago,2016-03-10 01:36:19,1
73.72,32,52140.04,256.4,Universal contextually-based system engine,Richardsland,1,Thailand,2016-03-18 02:39:26,0
38.1,34,60641.09,214.38,Optional secondary access,North Aaronchester,1,Philippines,2016-05-30 18:08:19,1
73.93,44,74180.05,218.22,Quality-focused scalable utilization,Lake Matthewland,0,Niue,2016-02-20 00:06:20,0
51.87,50,51869.87,119.65,Team-oriented dynamic forecast,Kevinberg,0,Afghanistan,2016-03-10 22:28:52,1
77.69,22,48852.58,169.88,Horizontal heuristic support,Morganfort,1,Angola,2016-06-21 14:32:32,0
43.41,28,59144.02,160.73,Customer-focused zero-defect process improvement,Lovemouth,0,Egypt,2016-02-05 15:26:37,1
55.92,24,33951.63,145.08,Focused systemic benchmark,Taylorhaven,0,Fiji,2016-05-31 21:41:46,1
80.67,34,58909.36,239.76,Seamless impactful info-mediaries,Jamesville,0,Portugal,2016-01-01 02:52:10,0
83.42,25,49850.52,183.42,Advanced heuristic firmware,East Toddfort,1,Austria,2016-03-04 14:10:12,0
82.12,52,28679.93,201.15,Fully-configurable client-driven customer loyalty,East Dana,1,Germany,2016-02-03 10:40:27,1
66.17,33,69869.66,238.45,Cross-group neutral synergy,West Lucas,0,Panama,2016-01-20 00:26:15,0
43.01,35,48347.64,127.37,Organized 24/7 middleware,Butlerfort,0,United States of America,2016-06-11 09:37:52,1
80.05,25,45959.86,219.94,Networked stable open architecture,Lindaside,1,Christmas Island,2016-03-08 05:48:20,0
64.88,42,70005.51,129.8,Customizable systematic service-desk,West Chloeborough,1,Equatorial Guinea,2016-02-14 22:23:30,1
79.82,26,51512.66,223.28,Function-based directional productivity,Jayville,1,Micronesia,2016-07-17 22:04:54,0
48.03,40,25598.75,134.6,Networked stable array,East Lindsey,1,Malta,2016-06-02 22:16:08,1
32.99,45,49282.87,177.46,Phased full-range hardware,Masseyshire,0,Ecuador,2016-04-30 19:42:04,1
74.88,27,67240.25,175.17,Organized empowering policy,Sarahton,1,Sudan,2016-04-17 06:58:18,0
36.49,52,42136.33,196.61,Object-based system-worthy superstructure,Ryanhaven,1,Lao People's Democratic Republic,2016-03-09 00:41:46,1
88.04,45,62589.84,191.17,Profound explicit hardware,Lake Deborahburgh,1,Saint Vincent and the Grenadines,2016-03-07 20:02:51,0
45.7,33,67384.31,151.12,Self-enabling multimedia system engine,New Williammouth,1,Switzerland,2016-05-26 10:33:00,1
82.38,35,25603.93,159.6,Polarized analyzing intranet,Port Blake,0,Spain,2016-07-18 01:36:37,1
52.68,23,39616.0,149.2,Vision-oriented attitude-oriented Internet solution,West Richard,1,Turks and Caicos Islands,2016-07-16 05:56:42,1
65.59,47,28265.81,121.81,Digitized disintermediate ability,Brandymouth,0,Indonesia,2016-03-22 06:41:38,1
65.65,25,63879.72,224.92,Intuitive explicit firmware,Sandraville,1,Cook Islands,2016-06-03 06:34:44,0
43.84,36,70592.81,167.42,Public-key real-time definition,Port Jessica,0,Australia,2016-06-28 09:19:06,1
67.69,37,76408.19,216.57,Monitored content-based implementation,Lake Jasonchester,0,Finland,2016-07-18 18:33:05,0
78.37,24,55015.08,207.27,Quality-focused zero-defect budgetary management,Pearsonfort,0,Pakistan,2016-01-23 04:47:37,0
81.46,29,51636.12,231.54,Intuitive fresh-thinking moderator,Sellerstown,0,Ireland,2016-02-29 11:00:06,0
47.48,31,29359.2,141.34,Reverse-engineered 24hour hardware,Yuton,0,Eritrea,2016-06-30 00:19:33,1
75.15,33,71296.67,219.49,Synchronized zero tolerance product,Smithtown,1,France,2016-06-19 18:19:38,0
78.76,24,46422.76,219.98,Reactive interactive protocol,Joanntown,1,Austria,2016-01-08 08:08:47,0
44.96,50,52802.0,132.71,Focused fresh-thinking Graphic Interface,South Peter,1,Heard Island and McDonald Islands,2016-01-02 12:25:36,1
39.56,41,59243.46,143.13,Ameliorated exuding solution,Port Mitchell,1,Western Sahara,2016-05-13 11:57:12,1
39.76,28,35350.55,196.83,Integrated maximized service-desk,Pottermouth,1,Liberia,2016-02-08 14:02:22,1
57.11,22,59677.64,207.17,Self-enabling tertiary challenge,Lake Jonathanview,1,Dominican Republic,2016-06-07 23:46:51,0
83.26,40,70225.6,187.76,Decentralized foreground infrastructure,Alanview,1,Tonga,2016-01-02 14:36:03,0
69.42,25,65791.17,213.38,Quality-focused hybrid frame,Carterport,0,Lao People's Democratic Republic,2016-02-13 04:16:08,0
50.6,30,34191.13,129.88,Realigned reciprocal framework,New Daniellefort,1,United States of America,2016-05-03 12:57:19,1
46.2,37,51315.38,119.3,Distributed maximized ability,Welchshire,0,Belgium,2016-04-03 11:38:36,1
66.88,35,62790.96,119.47,Polarized bifurcated array,Russellville,1,Indonesia,2016-03-23 19:58:15,1
83.97,40,66291.67,158.42,Progressive asynchronous adapter,West Lisa,1,Croatia,2016-02-02 11:49:18,0
76.56,30,68030.18,213.75,Business-focused high-level hardware,Greentown,0,Brunei Darussalam,2016-03-08 10:39:16,0
35.49,48,43974.49,159.77,Fully-configurable holistic throughput,Timothyport,0,American Samoa,2016-04-08 14:35:44,1
80.29,31,49457.48,244.87,Ameliorated contextually-based collaboration,Teresahaven,1,Netherlands Antilles,2016-06-30 00:40:31,0
50.19,40,33987.27,117.3,Progressive uniform budgetary management,Lake Stephenborough,0,Thailand,2016-03-25 19:02:35,1
59.12,33,28210.03,124.54,Synergistic stable infrastructure,Silvaton,0,Greece,2016-05-12 21:32:06,1
59.88,30,75535.14,193.63,Reverse-engineered content-based intranet,West Michaelstad,1,French Polynesia,2016-03-02 05:11:01,0
59.7,28,49158.5,120.25,Expanded zero administration attitude,Florestown,0,Guernsey,2016-05-10 14:12:31,1
67.8,30,39809.69,117.75,Team-oriented 6thgeneration extranet,New Jay,1,Isle of Man,2016-03-03 02:59:37,1
81.59,35,65826.53,223.16,Managed disintermediate capability,North Lisachester,0,Holy See (Vatican City State),2016-07-04 11:03:49,0
81.1,29,61172.07,216.49,Front-line dynamic model,Port Stacy,1,El Salvador,2016-07-08 03:47:41,0
41.7,39,42898.21,126.95,Innovative regional structure,Jensenton,0,China,2016-05-27 05:35:27,1
73.94,27,68333.01,173.49,Function-based incremental standardization,North Alexandra,0,Myanmar,2016-02-10 13:46:35,0
58.35,37,70232.95,132.63,Universal asymmetric workforce,Rivasland,0,Macao,2016-06-12 21:21:53,1
51.56,46,63102.19,124.85,Business-focused client-driven forecast,Helenborough,0,Australia,2016-01-07 13:58:51,1
79.81,37,51847.26,253.17,Realigned global initiative,Garnerberg,0,United States Virgin Islands,2016-05-13 14:12:39,0
66.17,26,63580.22,228.7,Business-focused maximized complexity,North Anaport,0,Mexico,2016-05-02 00:01:56,0
58.21,37,47575.44,105.94,Open-source global strategy,Pattymouth,0,Djibouti,2016-02-07 17:06:35,1
66.12,49,39031.89,113.8,Stand-alone motivating moratorium,South Alexisborough,0,Cote d'Ivoire,2016-02-15 07:27:41,1
80.47,42,70505.06,215.18,Grass-roots multimedia policy,East Jennifer,1,Mali,2016-02-21 05:23:28,0
77.05,31,62161.26,236.64,Upgradable local migration,Hallfort,0,Jamaica,2016-03-20 22:27:25,0
49.99,41,61068.26,121.07,Profound bottom-line standardization,New Charleschester,0,Romania,2016-03-24 09:34:00,1
80.3,58,49090.51,173.43,Managed client-server access,East Breannafurt,0,Cayman Islands,2016-04-04 20:01:12,1
79.36,33,62330.75,234.72,Cross-platform directional intranet,East Susanland,1,Gambia,2016-01-02 04:50:44,0
57.86,30,18819.34,166.86,Horizontal modular success,Estesfurt,0,Algeria,2016-07-08 17:14:01,1
70.29,26,62053.37,231.37,Vision-oriented multi-tasking success,Shirleyfort,1,Puerto Rico,2016-03-28 19:48:37,0
84.53,33,61922.06,215.18,Optional multi-state hardware,Douglasview,1,Norfolk Island,2016-07-11 09:32:53,0
59.13,44,49525.37,106.04,Upgradable heuristic system engine,South Lisa,1,Turkey,2016-06-09 17:11:02,1
81.51,41,53412.32,250.03,Future-proofed modular utilization,Kingshire,0,Guinea,2016-05-19 09:30:12,0
42.94,37,56681.65,130.4,Synergistic dynamic orchestration,Rebeccamouth,1,Moldova,2016-04-12 12:35:39,1
84.81,32,43299.63,233.93,Multi-layered stable encoding,Brownbury,1,Greece,2016-07-04 23:17:47,0
82.79,34,47997.75,132.08,Team-oriented zero-defect initiative,South Aaron,0,American Samoa,2016-02-01 00:52:29,1
59.22,55,39131.53,126.39,Polarized 5thgeneration matrix,North Andrew,1,Honduras,2016-01-13 02:39:00,1
35.0,40,46033.73,151.25,Fully-configurable context-sensitive Graphic Interface,South Walter,1,Mongolia,2016-06-18 16:02:34,1
46.61,42,65856.74,136.18,Progressive intermediate throughput,Catherinefort,0,Ethiopia,2016-01-01 20:17:49,1
63.26,29,54787.37,120.46,Customizable holistic archive,East Donna,1,Ethiopia,2016-03-02 04:02:45,1
79.16,32,69562.46,202.9,Compatible intermediate concept,East Timothy,1,Sri Lanka,2016-03-30 20:23:48,0
67.94,43,68447.17,128.16,Assimilated next generation firmware,North Kimberly,0,Morocco,2016-05-01 00:23:13,1
79.91,32,62772.42,230.18,Total zero administration software,South Stephanieport,1,United Arab Emirates,2016-06-17 03:02:55,0
66.14,41,78092.95,165.27,Re-engineered impactful software,North Isabellaville,0,Western Sahara,2016-03-23 08:52:31,0
43.65,39,63649.04,138.87,Business-focused background synergy,North Aaronburgh,0,Western Sahara,2016-05-08 22:24:27,1
59.61,21,60637.62,198.45,Future-proofed coherent budgetary management,Port James,1,Cambodia,2016-04-06 05:55:43,0
46.61,52,27241.11,156.99,Ergonomic methodical encoding,Danielview,0,New Zealand,2016-04-05 05:54:15,1
89.37,34,42760.22,162.03,Compatible dedicated productivity,Port Stacey,1,Australia,2016-04-16 12:26:31,0
65.1,49,59457.52,118.1,Up-sized real-time methodology,West Kevinfurt,1,Bulgaria,2016-06-01 03:44:42,1
53.44,42,42907.89,108.17,Up-sized next generation architecture,Lake Jennifer,1,Libyan Arab Jamahiriya,2016-04-04 22:00:15,1
79.53,51,46132.18,244.91,Managed 6thgeneration hierarchy,Reyesfurt,0,Barbados,2016-06-26 04:22:26,0
91.43,39,46964.11,209.91,Organic motivating model,West Carmenfurt,1,French Polynesia,2016-07-07 03:55:01,0
73.57,30,70377.23,212.38,Pre-emptive transitional protocol,North Stephanieberg,0,Uruguay,2016-03-20 08:22:50,0
78.76,32,70012.83,208.02,Managed attitude-oriented Internet solution,East Valerie,1,Uruguay,2016-04-20 10:04:29,0
76.49,23,56457.01,181.11,Public-key asynchronous matrix,Sherrishire,0,Brazil,2016-03-25 05:05:27,0
61.72,26,67279.06,218.49,Grass-roots systematic hardware,Port Daniel,0,Venezuela,2016-02-14 07:15:37,0
84.53,35,54773.99,236.29,User-centric composite contingency,Brownview,0,Myanmar,2016-03-26 00:32:02,0
72.03,34,70783.94,230.95,Up-sized bi-directional infrastructure,Greerton,1,Malta,2016-07-05 22:33:48,0
77.47,36,70510.59,222.91,Assimilated actuating policy,Hatfieldshire,1,Jamaica,2016-03-14 03:29:12,0
75.65,39,64021.55,247.9,Organized upward-trending contingency,Brianabury,1,Bahrain,2016-05-30 02:34:25,0
78.15,33,72042.85,194.37,Ergonomic neutral portal,New Maria,0,Algeria,2016-03-07 22:32:15,0
63.8,38,36037.33,108.7,Adaptive demand-driven knowledgebase,Colebury,1,Tuvalu,2016-03-19 00:27:58,1
76.59,29,67526.92,211.64,Reverse-engineered maximized focus group,Calebberg,0,Georgia,2016-06-18 05:17:33,0
42.6,55,55121.65,168.29,Switchable analyzing encryption,Lake Ian,0,Cambodia,2016-07-11 18:12:43,1
78.77,28,63497.62,211.83,Public-key intangible Graphical User Interface,Gomezport,0,Guam,2016-01-01 08:27:06,0
83.4,39,60879.48,235.01,Advanced local task-force,Shaneland,0,Tanzania,2016-04-07 01:57:38,0
79.53,33,61467.33,236.72,Profound well-modulated array,East Aaron,0,Indonesia,2016-02-28 22:02:14,0
73.89,35,70495.64,229.99,Multi-channeled asymmetric installation,Dustinborough,1,Somalia,2016-06-26 17:25:55,0
75.8,36,71222.4,224.9,Multi-layered fresh-thinking neural-net,East Michaelland,0,Belize,2016-01-21 04:30:43,0
81.95,31,64698.58,208.76,Distributed cohesive migration,East Connie,1,Serbia,2016-05-01 21:46:37,0
56.39,58,32252.38,154.23,Programmable uniform website,West Shannon,0,Australia,2016-02-14 10:06:49,1
44.73,35,55316.97,127.56,Object-based neutral policy,North Lauraland,1,Guam,2016-01-27 18:25:42,1
38.35,33,47447.89,145.48,Horizontal global leverage,Port Christopher,1,Christmas Island,2016-06-16 20:24:33,1
72.53,37,73474.82,223.93,Synchronized grid-enabled moratorium,South Patrickfort,0,Papua New Guinea,2016-07-21 10:01:50,0
56.2,49,53549.94,114.85,Adaptive uniform capability,East Georgeside,1,Bahamas,2016-04-21 18:31:27,1
79.67,28,58576.12,226.79,Total grid-enabled application,Charlesbury,0,Comoros,2016-07-20 01:56:33,0
75.42,26,63373.7,164.25,Optional regional throughput,Millertown,1,Western Sahara,2016-02-26 17:14:14,0
78.64,31,60283.47,235.28,Integrated client-server definition,South Renee,1,Nicaragua,2016-01-16 17:56:05,0
67.69,44,37345.34,109.22,Fundamental methodical support,South Jackieberg,0,Guam,2016-04-01 01:57:12,1
38.35,41,34886.01,144.69,Synergistic reciprocal attitude,Loriville,1,Vanuatu,2016-06-24 08:42:20,1
59.52,44,67511.86,251.08,Managed 5thgeneration time-frame,Amandaland,1,Bolivia,2016-05-27 18:45:35,0
62.26,37,77988.71,166.19,Vision-oriented uniform knowledgebase,West Robertside,0,Malawi,2016-05-26 15:40:12,0
64.75,36,63001.03,117.66,Multi-tiered stable leverage,North Sarashire,0,Venezuela,2016-04-06 01:19:08,1
79.97,26,61747.98,185.45,Down-sized explicit budgetary management,Port Maria,1,Nepal,2016-01-08 19:38:45,0
47.9,42,48467.68,114.53,Cross-group human-resource time-frame,East Jessefort,0,United Kingdom,2016-02-24 19:08:11,1
80.38,30,55130.96,238.06,Business-focused holistic benchmark,Port Anthony,0,Albania,2016-03-10 07:07:31,0
64.51,42,79484.8,190.71,Virtual 5thgeneration neural-net,Edwardmouth,1,Madagascar,2016-04-29 07:49:01,0
71.28,37,67307.43,246.72,Distributed scalable orchestration,Dustinchester,1,Guyana,2016-04-10 16:08:09,0
50.32,40,27964.6,125.65,Realigned intangible benchmark,Rochabury,0,Yemen,2016-04-27 18:25:30,1
72.76,33,66431.87,240.63,Virtual impactful algorithm,Williamsport,1,India,2016-05-10 04:28:55,0
72.8,35,63551.67,249.54,Public-key solution-oriented focus group,Austinland,0,Puerto Rico,2016-01-03 23:21:26,0
74.59,23,40135.06,158.35,Phased clear-thinking encoding,Lake Gerald,1,United States Virgin Islands,2016-02-15 16:52:04,1
46.66,45,49101.67,118.16,Grass-roots mission-critical emulation,Wrightview,0,Antigua and Barbuda,2016-03-09 02:07:17,1
48.86,54,53188.69,134.46,Proactive encompassing paradigm,Perryburgh,0,French Guiana,2016-01-09 17:33:03,1
37.05,39,49742.83,142.81,Automated object-oriented firmware,Tracyhaven,1,Antigua and Barbuda,2016-02-03 05:47:09,1
81.21,36,63394.41,233.04,User-friendly content-based customer loyalty,South Jaimeview,0,Turkmenistan,2016-01-02 09:30:11,0
66.89,23,64433.99,208.24,Universal incremental array,Sandersland,1,Honduras,2016-01-04 07:28:43,0
68.11,38,73884.48,231.21,Reactive national success,South Meredithmouth,0,Seychelles,2016-01-07 21:21:50,0
69.15,46,36424.94,112.72,Automated multi-state toolset,Richardsonshire,0,Cyprus,2016-07-24 00:22:16,1
65.72,36,28275.48,120.12,Managed didactic flexibility,Kimberlymouth,0,Saint Pierre and Miquelon,2016-02-13 13:57:53,1
40.04,27,48098.86,161.58,Cross-platform neutral system engine,Meghanchester,0,Poland,2016-05-08 10:25:08,1
68.6,33,68448.94,135.08,Focused high-level frame,Tammyshire,0,Taiwan,2016-02-17 18:50:57,1
56.16,25,66429.84,164.25,Seamless motivating approach,Millerbury,1,Cote d'Ivoire,2016-01-22 19:43:53,1
78.6,46,41768.13,254.59,Enhanced systematic adapter,Lake Elizabethside,1,Micronesia,2016-07-20 13:21:37,0
78.29,38,57844.96,252.07,Networked regional Local Area Network,Villanuevaton,0,Liberia,2016-01-05 20:58:42,0
43.83,45,35684.82,129.01,Total human-resource flexibility,Greerport,0,Saudi Arabia,2016-01-29 05:39:16,1
77.31,32,62792.43,238.1,Assimilated homogeneous service-desk,North Garyhaven,0,Nepal,2016-06-17 20:18:27,0
39.86,28,51171.23,161.24,Ergonomic zero tolerance encoding,East Sharon,0,Ghana,2016-02-23 13:55:48,1
66.77,25,58847.07,141.13,Cross-platform zero-defect structure,Johnstonmouth,0,Iran,2016-07-09 11:18:02,1
57.2,42,57739.03,110.66,Innovative maximized groupware,East Heatherside,0,New Zealand,2016-03-19 11:09:36,1
73.15,25,64631.22,211.12,Face-to-face executive encryption,Lake Patrick,1,Libyan Arab Jamahiriya,2016-01-29 07:14:04,0
82.07,24,50337.93,193.97,Monitored local Internet solution,Richardsonmouth,0,Sri Lanka,2016-06-14 07:02:09,0
49.84,38,67781.31,135.24,Phased hybrid superstructure,Jenniferhaven,1,United Arab Emirates,2016-05-18 03:19:03,1
43.97,36,68863.95,156.97,User-friendly grid-enabled analyzer,Boyerberg,1,Indonesia,2016-01-30 09:54:03,1
77.25,27,55901.12,231.38,Pre-emptive neutral contingency,Port Elijah,1,Saint Vincent and the Grenadines,2016-04-25 16:58:50,0
74.84,37,64775.1,246.44,User-friendly impactful time-frame,Knappburgh,1,Mongolia,2016-01-14 16:30:38,0
83.53,36,67686.16,204.56,Customizable methodical Graphical User Interface,New Dawnland,0,Honduras,2016-07-06 05:34:52,0
38.63,48,57777.11,222.11,Cross-platform logistical pricing structure,Chapmanmouth,0,Papua New Guinea,2016-04-07 10:51:05,1
84.0,48,46868.53,136.21,Inverse discrete extranet,Robertside,1,Kyrgyz Republic,2016-04-17 05:08:52,1
52.13,50,40926.93,118.27,Open-source even-keeled database,West Raymondmouth,1,Ethiopia,2016-01-28 17:03:54,1
71.83,40,22205.74,135.48,Diverse background ability,Costaburgh,1,Rwanda,2016-02-18 22:42:33,1
78.36,24,58920.44,196.77,Multi-tiered foreground Graphic Interface,Kristineberg,1,Kyrgyz Republic,2016-06-24 21:09:58,0
50.18,35,63006.14,127.82,Customizable hybrid system engine,Sandrashire,1,Grenada,2016-06-20 04:24:41,1
64.67,51,24316.61,138.35,Horizontal incremental website,Andersonfurt,1,Togo,2016-02-14 16:33:29,1
69.5,26,68348.99,203.84,Front-line systemic capability,Tranland,0,Pakistan,2016-02-27 13:51:44,0
65.22,30,66263.37,240.09,Fully-configurable foreground solution,Michaelland,1,Falkland Islands (Malvinas),2016-05-07 15:16:07,0
62.06,40,63493.6,116.27,Digitized radical array,East Rachaelfurt,1,Jersey,2016-03-16 20:10:53,1
84.29,30,56984.09,160.33,Team-oriented transitional methodology,Lake Johnbury,1,Cayman Islands,2016-06-26 02:06:59,1
32.91,37,51691.55,181.02,Future-proofed fresh-thinking conglomeration,Elizabethstad,0,South Africa,2016-07-17 14:26:04,1
39.5,31,49911.25,148.19,Operative multi-tasking Graphic Interface,West Brad,1,Micronesia,2016-01-28 16:42:36,1
75.19,31,33502.57,245.76,Implemented discrete frame,Johnstonshire,1,Tajikistan,2016-06-16 18:04:51,0
76.21,31,65834.97,228.94,Ameliorated exuding encryption,Lake Timothy,1,Bolivia,2016-06-19 23:21:38,0
67.76,31,66176.97,242.59,Programmable high-level benchmark,Anthonyfurt,0,Cameroon,2016-05-24 17:42:58,0
40.01,53,51463.17,161.77,Sharable multimedia conglomeration,East Brettton,0,Ecuador,2016-03-01 22:06:37,1
52.7,41,41059.64,109.34,Team-oriented high-level orchestration,New Matthew,1,Zambia,2016-01-31 08:50:38,1
68.41,38,61428.18,259.76,Grass-roots empowering paradigm,Christopherchester,0,Guinea-Bissau,2016-04-30 15:27:22,0
35.55,39,51593.46,151.18,Robust object-oriented Graphic Interface,Westshire,0,Micronesia,2016-01-13 20:38:35,1
74.54,24,57518.73,219.75,Switchable secondary ability,Alexisland,0,Bahamas,2016-03-30 16:15:59,0
81.75,24,52656.13,190.08,Open-architected web-enabled benchmark,Kevinchester,1,Cape Verde,2016-04-29 18:53:43,0
87.85,31,52178.98,210.27,Compatible scalable emulation,New Patriciashire,1,French Polynesia,2016-06-14 19:48:34,0
60.23,60,46239.14,151.54,Seamless optimal contingency,Port Brenda,1,Saudi Arabia,2016-07-15 15:43:36,1
87.97,35,48918.55,149.25,Secured secondary superstructure,Port Brianfort,1,France,2016-03-24 05:38:01,0
78.17,27,65227.79,192.27,Automated mobile model,Portermouth,1,Burundi,2016-04-26 20:57:48,0
67.91,23,55002.05,146.8,Re-engineered non-volatile neural-net,Hubbardmouth,1,Latvia,2016-01-12 03:28:31,1
85.77,27,52261.73,191.78,Implemented disintermediate attitude,South Brian,1,Morocco,2016-04-09 23:26:42,0
41.16,49,59448.44,150.83,Configurable interactive contingency,Hendrixmouth,1,Venezuela,2016-03-28 09:15:58,1
53.54,39,47314.45,108.03,Optimized systemic capability,Julietown,0,Palau,2016-06-23 11:05:01,1
73.94,26,55411.06,236.15,Front-line non-volatile implementation,Lukeport,1,Isle of Man,2016-01-24 01:53:14,0
63.43,29,66504.16,236.75,Ergonomic 24/7 solution,New Shane,1,Peru,2016-04-15 10:18:55,0
84.59,36,47169.14,241.8,Integrated grid-enabled budgetary management,Lake Jillville,1,Belgium,2016-04-26 13:13:20,0
70.13,31,70889.68,224.98,Profit-focused systemic support,Johnsonfort,0,Croatia,2016-05-16 23:21:06,0
40.19,37,55358.88,136.99,Right-sized system-worthy project,Adamsbury,0,France,2016-01-18 02:51:13,1
58.95,55,56242.7,131.29,Proactive actuating Graphical User Interface,East Maureen,1,Slovenia,2016-06-20 08:34:46,1
35.76,51,45522.44,195.07,Versatile optimizing projection,North Angelastad,0,Peru,2016-07-18 04:53:22,1
59.36,49,46931.03,110.84,Universal multi-state system engine,Amandafort,0,Belarus,2016-07-01 01:12:04,1
91.1,40,55499.69,198.13,Secured intermediate approach,Michaelmouth,1,Bolivia,2016-03-07 22:51:00,0
61.04,41,75805.12,149.21,Operative didactic Local Area Network,Ronaldport,0,Benin,2016-05-02 15:31:28,1
74.06,23,40345.49,225.99,Phased content-based middleware,Port Davidland,0,Wallis and Futuna,2016-07-23 06:18:51,0
64.63,45,15598.29,158.8,Triple-buffered high-level Internet solution,Isaacborough,1,Azerbaijan,2016-06-12 03:11:04,1
81.29,28,33239.2,219.72,Synergized well-modulated Graphical User Interface,Lake Michael,0,Mongolia,2016-02-15 20:41:05,0
76.07,36,68033.54,235.56,Implemented bottom-line implementation,West Michaelshire,0,Denmark,2016-01-23 01:42:28,0
75.92,22,38427.66,182.65,Monitored context-sensitive initiative,Port Calvintown,0,Russian Federation,2016-02-26 01:18:44,0
78.35,46,53185.34,253.48,Pre-emptive client-server open system,Parkerhaven,0,Brazil,2016-01-11 02:07:14,0
46.14,28,39723.97,137.97,Seamless bandwidth-monitored knowledge user,Markhaven,1,Ethiopia,2016-04-04 13:56:14,1
44.33,41,43386.07,120.63,Ergonomic empowering frame,Estradashire,0,Guyana,2016-01-14 09:27:59,1
46.43,28,53922.43,137.2,Reverse-engineered background Graphic Interface,Brianland,1,Ethiopia,2016-04-25 03:18:45,1
66.04,27,71881.84,199.76,Synergistic non-volatile analyzer,Cassandratown,0,Mauritius,2016-03-05 23:02:11,0
84.31,29,47139.21,225.87,Object-based optimal solution,West Dannyberg,0,Djibouti,2016-01-06 21:43:22,0
83.66,38,68877.02,175.14,Profound dynamic attitude,East Debraborough,0,Syrian Arab Republic,2016-02-18 03:58:36,0
81.25,33,65186.58,222.35,Enhanced system-worthy toolset,Frankchester,1,Saint Martin,2016-04-16 14:15:55,0
85.26,32,55424.24,224.07,Reverse-engineered dynamic function,Lisafort,1,Netherlands Antilles,2016-02-24 06:18:11,0
86.53,46,46500.11,233.36,Networked responsive application,Colemanshire,0,Greece,2016-06-29 01:19:21,0
76.44,26,58820.16,224.2,Distributed intangible database,Troyville,1,Madagascar,2016-01-05 06:34:20,0
52.84,43,28495.21,122.31,Multi-tiered mobile encoding,Hobbsbury,0,Senegal,2016-07-16 10:14:04,1
85.24,31,61840.26,182.84,Optional contextually-based flexibility,Harrisonmouth,1,Burkina Faso,2016-06-17 03:23:13,0
74.71,46,37908.29,258.06,Proactive local focus group,Port Eugeneport,1,Czech Republic,2016-06-13 11:06:40,0
82.95,39,69805.7,201.29,Customer-focused impactful success,Karenmouth,0,Lao People's Democratic Republic,2016-04-05 08:18:45,0
76.42,26,60315.19,223.16,Open-source optimizing parallelism,Brendaburgh,1,Netherlands Antilles,2016-04-17 18:38:14,0
42.04,49,67323.0,182.11,Organic logistical adapter,New Christinatown,0,Qatar,2016-02-03 16:54:33,1
46.28,26,50055.33,228.78,Stand-alone eco-centric system engine,Jacksonstad,1,Andorra,2016-04-18 21:07:28,1
48.26,50,43573.66,122.45,User-centric intermediate knowledge user,South Margaret,1,Liechtenstein,2016-06-18 22:31:22,1
71.03,55,28186.65,150.77,Programmable didactic capacity,Port Georgebury,0,China,2016-03-12 07:18:36,1
81.37,33,66412.04,215.04,Enhanced regional conglomeration,New Jessicaport,0,Vietnam,2016-01-15 01:20:05,0
58.05,32,15879.1,195.54,Total asynchronous architecture,Sanderstown,1,Tajikistan,2016-02-12 10:39:10,1
75.0,29,63965.16,230.36,Secured upward-trending benchmark,Perezland,1,Eritrea,2016-02-16 02:29:03,0
79.61,31,58342.63,235.97,Customizable value-added project,Luisfurt,0,Monaco,2016-04-04 21:23:13,0
52.56,31,33147.19,250.36,Integrated interactive support,New Karenberg,1,Israel,2016-04-24 01:48:21,1
62.18,33,65899.68,126.44,Reactive impactful challenge,West Leahton,0,Hungary,2016-05-20 00:00:48,1
77.89,26,64188.5,201.54,Switchable multi-state success,West Sharon,0,Singapore,2016-05-15 03:10:50,0
66.08,61,58966.22,184.23,Synchronized multi-tasking ability,Klineside,1,Cuba,2016-01-07 23:02:43,1
89.21,33,44078.24,210.53,Fundamental clear-thinking knowledgebase,Lake Cynthia,0,Reunion,2016-07-19 12:05:58,0
49.96,55,60968.62,151.94,Multi-layered user-facing parallelism,South Cynthiashire,1,Zambia,2016-04-04 00:02:20,1
77.44,28,65620.25,210.39,Front-line incremental access,Lake Jacob,0,Gabon,2016-06-10 04:21:57,0
82.58,38,65496.78,225.23,Open-architected zero administration secured line,West Samantha,1,Dominica,2016-03-11 14:50:56,0
39.36,29,52462.04,161.79,Mandatory disintermediate info-mediaries,Jeremybury,1,Bahamas,2016-01-14 20:58:10,1
47.23,38,70582.55,149.8,Implemented context-sensitive Local Area Network,Blevinstown,1,Tokelau,2016-06-22 05:22:58,1
87.85,34,51816.27,153.01,Digitized interactive initiative,Meyerchester,0,Turkmenistan,2016-03-19 08:00:58,0
65.57,46,23410.75,130.86,Implemented asynchronous application,Reginamouth,0,Belgium,2016-04-15 15:07:17,1
78.01,26,62729.4,200.71,Focused multi-state workforce,Donaldshire,1,French Guiana,2016-03-28 02:29:19,0
44.15,28,48867.67,141.96,Proactive secondary monitoring,Salazarbury,1,Martinique,2016-01-22 15:03:25,1
43.57,36,50971.73,125.2,Front-line upward-trending groupware,Lake Joshuafurt,1,French Polynesia,2016-06-25 17:33:35,1
76.83,28,67990.84,192.81,Quality-focused 5thgeneration orchestration,Wintersfort,0,Ecuador,2016-03-04 14:33:38,0
42.06,34,43241.19,131.55,Multi-layered secondary software,Jamesmouth,0,Puerto Rico,2016-06-29 02:48:44,1
76.27,27,60082.66,226.69,Total coherent superstructure,Laurieside,1,United Arab Emirates,2016-06-18 01:42:37,0
74.27,37,65180.97,247.05,Monitored executive architecture,Andrewmouth,1,Burkina Faso,2016-01-31 09:57:34,0
73.27,28,67301.39,216.24,Front-line multi-state hub,West Angela,1,Luxembourg,2016-05-22 15:17:25,0
74.58,36,70701.31,230.52,Configurable mission-critical algorithm,East Carlos,0,Jamaica,2016-07-22 11:05:10,0
77.5,28,60997.84,225.34,Face-to-face responsive alliance,Kennedyfurt,1,Antarctica (the territory South of 60 deg S),2016-07-13 14:05:22,0
87.16,33,60805.93,197.15,Reduced holistic help-desk,Blairville,0,China,2016-02-11 11:50:26,0
87.16,37,50711.68,231.95,Pre-emptive content-based frame,East Donnatown,1,Western Sahara,2016-03-16 20:33:10,0
66.26,47,14548.06,179.04,Optional full-range projection,Matthewtown,1,Lebanon,2016-04-25 19:31:39,1
65.15,29,41335.84,117.3,Expanded value-added emulation,Brandonbury,0,Hong Kong,2016-07-14 22:43:29,1
68.25,33,76480.16,198.86,Organic well-modulated database,New Jamestown,1,Vanuatu,2016-05-30 08:02:35,0
73.49,38,67132.46,244.23,Organic 3rdgeneration encryption,Mosleyburgh,0,Vanuatu,2016-02-14 11:36:08,0
39.19,54,52581.16,173.05,Stand-alone empowering benchmark,Leahside,0,Guatemala,2016-01-23 21:15:57,1
80.15,25,55195.61,214.49,Monitored intermediate circuit,West Wendyland,0,Greenland,2016-07-18 02:51:19,0
86.76,28,48679.54,189.91,Object-based leadingedge complexity,Lawrenceborough,0,Syrian Arab Republic,2016-02-10 08:21:13,0
73.88,29,63109.74,233.61,Digitized zero-defect implementation,Kennethview,0,Saint Helena,2016-01-04 06:37:15,0
58.6,19,44490.09,197.93,Configurable impactful firmware,West Mariafort,1,Lebanon,2016-06-05 21:38:22,0
69.77,54,57667.99,132.27,Face-to-face dedicated flexibility,Port Sherrystad,0,Malta,2016-06-01 03:17:50,1
87.27,30,51824.01,204.27,Fully-configurable 5thgeneration circuit,West Melissashire,1,Christmas Island,2016-03-06 06:51:23,1
77.65,28,66198.66,208.01,Configurable impactful capacity,Pamelamouth,0,Ukraine,2016-02-26 19:35:54,0
76.02,40,73174.19,219.55,Distributed leadingedge orchestration,Lesliefort,0,Malta,2016-07-13 14:30:14,0
78.84,26,56593.8,217.66,Persistent even-keeled application,Shawnside,1,Italy,2016-06-29 07:20:46,0
71.33,23,31072.44,169.4,Optimized attitude-oriented initiative,Josephmouth,0,Japan,2016-03-15 06:54:21,1
81.9,41,66773.83,225.47,Multi-channeled 3rdgeneration model,Garciatown,0,Mauritius,2016-06-11 06:47:55,0
46.89,48,72553.94,176.78,Polarized mission-critical structure,Chaseshire,1,Turkey,2016-07-17 13:22:43,1
77.8,57,43708.88,152.94,Virtual executive implementation,Destinyfurt,0,Namibia,2016-02-14 14:38:01,1
45.44,43,48453.55,119.27,Enhanced intermediate standardization,Mezaton,0,China,2016-05-04 05:01:37,1
69.96,31,73413.87,214.06,Realigned tangible collaboration,New Kayla,1,Netherlands,2016-05-20 12:17:28,0
87.35,35,58114.3,158.29,Cloned dedicated analyzer,Carsonshire,1,Gibraltar,2016-01-26 02:47:17,0
49.42,53,45465.25,128.0,Ameliorated well-modulated complexity,Jacquelineshire,1,Congo,2016-07-07 18:07:19,1
71.27,21,50147.72,216.03,Quality-focused bi-directional throughput,South Blakestad,1,Senegal,2016-01-11 12:46:31,0
49.19,38,61004.51,123.08,Versatile solution-oriented secured line,North Mark,0,Hungary,2016-05-12 12:11:12,1
39.96,35,53898.89,138.52,Phased leadingedge budgetary management,Kingchester,1,Pitcairn Islands,2016-02-28 23:21:22,1
85.01,29,59797.64,192.5,Devolved exuding Local Area Network,Evansfurt,0,Slovakia (Slovak Republic),2016-05-03 16:02:50,0
68.95,51,74623.27,185.85,Front-line bandwidth-monitored capacity,South Adamhaven,1,United States Virgin Islands,2016-03-15 20:19:20,0
67.59,45,58677.69,113.69,User-centric solution-oriented emulation,Brittanyborough,0,Monaco,2016-07-23 05:21:39,1
75.71,34,62109.8,246.06,Phased hybrid intranet,Barbershire,0,Portugal,2016-03-11 10:01:23,0
43.07,36,60583.02,137.63,Monitored zero administration collaboration,East Ericport,1,Turkey,2016-02-11 20:45:46,1
39.47,43,65576.05,163.48,Team-oriented systematic installation,Crawfordfurt,1,Uganda,2016-07-06 23:09:07,1
48.22,40,73882.91,214.33,Inverse national core,Turnerville,0,Norfolk Island,2016-03-22 19:14:47,0
76.76,25,50468.36,230.77,Secured uniform instruction set,Kylieview,1,Niue,2016-05-26 13:28:36,0
78.74,27,51409.45,234.75,Quality-focused zero tolerance matrices,West Zacharyborough,0,Ukraine,2016-06-18 19:10:14,0
67.47,24,60514.05,225.05,Multi-tiered heuristic strategy,Watsonfort,1,Vanuatu,2016-03-20 07:12:52,0
81.17,30,57195.96,231.91,Optimized static archive,Dayton,1,United States Minor Outlying Islands,2016-06-03 07:00:36,0
89.66,34,52802.58,171.23,Advanced didactic conglomeration,Nicholasport,1,Armenia,2016-02-03 15:15:42,0
79.6,28,56570.06,227.37,Synergistic discrete middleware,Whitneyfort,1,Sweden,2016-05-03 16:55:02,0
65.53,19,51049.47,190.17,Pre-emptive client-server installation,Coffeytown,1,Timor-Leste,2016-06-20 02:25:12,0
61.87,35,66629.61,250.2,Multi-channeled attitude-oriented toolset,North Johnside,1,French Southern Territories,2016-07-10 19:15:52,0
83.16,41,70185.06,194.95,Decentralized 24hour approach,Robinsonland,0,Finland,2016-01-04 04:00:35,0
44.11,41,43111.41,121.24,Organic next generation matrix,Lake David,1,Saint Vincent and the Grenadines,2016-04-20 16:49:15,1
56.57,26,56435.6,131.98,Multi-channeled non-volatile website,West Ericaport,0,Senegal,2016-01-23 13:14:18,1
83.91,29,53223.58,222.87,Distributed bifurcated challenge,Haleberg,0,Burundi,2016-01-04 22:27:25,0
79.8,28,57179.91,229.88,Customizable zero-defect Internet solution,West Michaelport,1,Bahamas,2016-04-08 22:40:55,0
71.23,52,41521.28,122.59,Self-enabling zero administration neural-net,Ericksonmouth,0,Sweden,2016-01-05 11:53:17,1
47.23,43,73538.09,210.87,Optimized upward-trending productivity,Yangside,1,Svalbard & Jan Mayen Islands,2016-03-17 22:24:02,1
82.37,30,63664.32,207.44,Open-architected system-worthy ability,Estradafurt,0,Tonga,2016-06-29 04:23:10,0
43.63,38,61757.12,135.25,Quality-focused maximized extranet,Frankport,1,Korea,2016-05-25 19:45:16,1
70.9,28,71727.51,190.95,Centralized client-driven workforce,Port Juan,0,Kyrgyz Republic,2016-06-17 23:19:38,0
71.9,29,72203.96,193.29,De-engineered intangible flexibility,Williamsside,1,Costa Rica,2016-04-24 07:20:16,0
62.12,37,50671.6,105.86,Re-engineered intangible software,Johnsonview,1,Liechtenstein,2016-03-18 13:00:12,1
67.35,29,47510.42,118.69,Sharable secondary Graphical User Interface,East Heidi,0,Zimbabwe,2016-04-28 21:58:25,1
57.99,50,62466.1,124.58,Innovative homogeneous alliance,New Angelview,0,Costa Rica,2016-02-12 08:46:15,1
66.8,29,59683.16,248.51,Diverse leadingedge website,Lake Brandonview,0,Hungary,2016-07-11 13:23:37,1
49.13,32,41097.17,120.49,Optimized intermediate help-desk,Morganport,0,Fiji,2016-01-29 00:45:19,1
45.11,58,39799.73,195.69,Sharable reciprocal project,Browntown,0,Netherlands,2016-01-05 16:26:44,1
54.35,42,76984.21,164.02,Proactive interactive service-desk,Lake Hailey,0,Sweden,2016-06-20 08:22:09,0
61.82,59,57877.15,151.93,Open-architected needs-based customer loyalty,Olsonside,1,Barbados,2016-02-06 17:48:28,1
77.75,31,59047.91,240.64,Multi-lateral motivating circuit,Coxhaven,1,Paraguay,2016-06-22 17:19:09,0
70.61,28,72154.68,190.12,Assimilated encompassing portal,Meaganfort,0,Italy,2016-04-16 05:24:33,0
82.72,31,65704.79,179.82,Cross-group global orchestration,North Monicaville,0,Belarus,2016-01-17 05:07:11,0
76.87,36,72948.76,212.59,Down-sized bandwidth-monitored core,Mullenside,0,South Georgia and the South Sandwich Islands,2016-07-08 22:30:10,0
65.07,34,73941.91,227.53,Monitored explicit hierarchy,Princebury,1,Anguilla,2016-03-11 00:05:48,0
56.93,37,57887.64,111.8,Reactive demand-driven strategy,Bradleyside,0,Sierra Leone,2016-06-10 00:35:15,1
48.86,35,62463.7,128.37,Universal empowering adapter,Elizabethbury,1,Saint Martin,2016-01-04 00:44:57,1
36.56,29,42838.29,195.89,Team-oriented bi-directional secured line,West Ryan,0,Uganda,2016-01-01 15:14:24,1
85.73,32,43778.88,147.75,Stand-alone radical throughput,New Tammy,1,Saudi Arabia,2016-07-10 17:24:51,1
75.81,40,71157.05,229.19,Inverse zero-defect capability,Sanchezland,0,Greenland,2016-03-27 19:50:11,0
72.94,31,74159.69,190.84,Multi-tiered real-time implementation,Rogerland,0,Venezuela,2016-04-29 13:38:19,0
53.63,54,50333.72,126.29,Front-line zero-defect array,Vanessaview,1,Liberia,2016-01-08 18:13:43,1
52.35,25,33293.78,147.61,Mandatory 4thgeneration structure,Jessicashire,1,Mali,2016-06-05 07:54:30,1
52.84,51,38641.2,121.57,Synergistic asynchronous superstructure,Melissachester,1,Bosnia and Herzegovina,2016-06-29 10:50:45,1
51.58,33,49822.78,115.91,Vision-oriented system-worthy forecast,Johnsontown,0,Brunei Darussalam,2016-04-24 13:46:10,1
42.32,29,63891.29,187.09,Digitized radical architecture,New Joshuaport,1,South Georgia and the South Sandwich Islands,2016-02-14 04:14:13,1
55.04,42,43881.73,106.96,Quality-focused optimizing parallelism,Hernandezside,1,Czech Republic,2016-06-15 05:43:02,1
68.58,41,13996.5,171.54,Exclusive discrete firmware,New Williamville,1,El Salvador,2016-07-06 12:04:29,1
85.54,27,48761.14,175.43,Right-sized solution-oriented benchmark,Gilbertville,1,Tokelau,2016-03-31 13:54:51,0
71.14,30,69758.31,224.82,Assimilated stable encryption,Newmanberg,0,France,2016-06-21 00:52:47,0
64.38,19,52530.1,180.47,Configurable dynamic secured line,West Alice,1,Gabon,2016-05-27 05:23:26,0
88.85,40,58363.12,213.96,Cloned optimal leverage,Cannonbury,0,Bulgaria,2016-01-17 18:45:55,0
66.79,60,60575.99,198.3,Decentralized client-driven data-warehouse,Shelbyport,1,Burkina Faso,2016-04-07 20:34:42,1
32.6,45,48206.04,185.47,Multi-tiered interactive neural-net,New Henry,0,Mayotte,2016-05-02 18:37:01,1
43.88,54,31523.09,166.85,Enhanced methodical database,Dustinmouth,1,Somalia,2016-06-04 17:24:07,1
56.46,26,66187.58,151.63,Ameliorated leadingedge help-desk,South Lisa,0,Albania,2016-04-07 18:52:57,1
72.18,30,69438.04,225.02,De-engineered attitude-oriented projection,Lisamouth,0,Bolivia,2016-06-10 22:21:10,0
52.67,44,14775.5,191.26,Persevering 5thgeneration knowledge user,New Hollyberg,0,Jersey,2016-05-19 06:37:38,1
80.55,35,68016.9,219.91,Extended grid-enabled hierarchy,Port Brittanyville,0,British Virgin Islands,2016-03-28 23:01:24,0
67.85,41,78520.99,202.7,Reactive tangible contingency,East Ronald,1,Saint Helena,2016-01-21 22:51:34,1
75.55,36,31998.72,123.71,Decentralized attitude-oriented interface,South Davidmouth,1,Bosnia and Herzegovina,2016-03-12 06:05:12,1
80.46,29,56909.3,230.78,Mandatory coherent groupware,Carterton,0,India,2016-06-04 09:13:29,0
82.69,29,61161.29,167.41,Fully-configurable eco-centric frame,Rachelhaven,1,Georgia,2016-05-24 10:16:38,0
35.21,39,52340.1,154.0,Advanced disintermediate data-warehouse,New Timothy,1,United States Minor Outlying Islands,2016-03-25 06:36:53,1
36.37,40,47338.94,144.53,Quality-focused zero-defect data-warehouse,North Jessicaville,1,Kiribati,2016-04-22 00:28:18,1
74.07,22,50950.24,165.43,Cross-group non-volatile secured line,Joneston,1,Ghana,2016-03-22 04:13:35,0
59.96,33,77143.61,197.66,Expanded modular application,Staceyfort,0,Samoa,2016-01-14 08:27:04,1
85.62,29,57032.36,195.68,Triple-buffered systematic info-mediaries,South Dianeshire,0,Iran,2016-04-14 21:37:49,0
40.88,33,48554.45,136.18,Networked non-volatile synergy,West Shannon,1,Costa Rica,2016-05-31 17:50:15,1
36.98,31,39552.49,167.87,Fully-configurable clear-thinking throughput,Micheletown,1,Northern Mariana Islands,2016-03-17 06:25:47,1
35.49,47,36884.23,170.04,Front-line actuating functionalities,North Brittanyburgh,0,Liechtenstein,2016-04-13 07:07:36,1
56.56,26,68783.45,204.47,Compatible composite project,Port Jasmine,1,Grenada,2016-02-03 22:11:13,0
36.62,32,51119.93,162.44,Customer-focused solution-oriented software,New Sabrina,1,Poland,2016-02-02 19:59:17,1
49.35,49,44304.13,119.86,Inverse stable synergy,Lake Charlottestad,0,Kenya,2016-04-07 20:38:02,1
75.64,29,69718.19,204.82,Pre-emptive well-modulated moderator,West Rhondamouth,1,Iran,2016-03-15 19:35:19,0
79.22,27,63429.18,198.79,Intuitive modular system engine,North Debra,1,Belgium,2016-03-11 12:39:19,0
77.05,34,65756.36,236.08,Centralized value-added hierarchy,Villanuevastad,0,Namibia,2016-05-17 18:06:46,0
66.83,46,77871.75,196.17,Assimilated hybrid initiative,North Jeremyport,1,Cyprus,2016-02-28 23:10:32,0
76.2,24,47258.59,228.81,Optimized coherent Internet solution,Lake Susan,1,Japan,2016-03-02 06:35:08,0
56.64,29,55984.89,123.24,Versatile 6thgeneration parallelism,Lake John,1,Zimbabwe,2016-02-27 08:52:50,1
53.33,34,44275.13,111.63,Configurable impactful productivity,Courtneyfort,1,Andorra,2016-03-14 04:34:35,1
50.63,50,25767.16,142.23,Operative full-range forecast,Tammymouth,0,Luxembourg,2016-03-10 15:07:44,1
41.84,49,37605.11,139.32,Operative secondary functionalities,Lake Vanessa,0,Cyprus,2016-05-01 08:27:12,1
53.92,41,25739.09,125.46,Business-focused transitional solution,Lake Amanda,1,Turkey,2016-06-12 11:17:25,1
83.89,28,60188.38,180.88,Ameliorated intermediate Graphical User Interface,Mariemouth,1,Hong Kong,2016-05-28 12:20:15,0
55.32,43,67682.32,127.65,Managed 24hour analyzer,Port Douglasborough,0,Netherlands,2016-03-18 09:08:39,1
53.22,44,44307.18,108.85,Horizontal client-server database,Port Aprilville,0,United States Virgin Islands,2016-05-26 06:03:57,1
43.16,35,25371.52,156.11,Implemented didactic support,Williamsport,1,Marshall Islands,2016-07-06 03:40:17,1
67.51,43,23942.61,127.2,Digitized homogeneous core,Lake Faith,0,Western Sahara,2016-04-29 14:10:00,1
43.16,29,50666.5,143.04,Robust holistic application,Wendyville,1,Saint Vincent and the Grenadines,2016-03-05 20:53:19,1
79.89,30,50356.06,241.38,Synergized uniform hierarchy,Angelhaven,1,United States of America,2016-05-30 08:35:54,0
84.25,32,63936.5,170.9,Pre-emptive client-driven secured line,New Sean,1,Angola,2016-04-10 06:32:11,0
74.18,28,69874.18,203.87,Front-line even-keeled website,Lake Lisa,0,Cayman Islands,2016-01-20 02:31:36,0
85.78,34,50038.65,232.78,Persistent fault-tolerant service-desk,Valerieland,0,Swaziland,2016-07-20 21:53:42,0
80.96,39,67866.95,225.0,Integrated leadingedge frame,New Travis,1,Wallis and Futuna,2016-01-17 04:12:30,0
36.91,48,54645.2,159.69,Ameliorated coherent open architecture,North Samantha,0,Zimbabwe,2016-02-24 07:13:00,1
54.47,23,46780.09,141.52,Vision-oriented bifurcated contingency,Holderville,0,Chad,2016-03-26 19:37:46,1
81.98,34,67432.49,212.88,Up-sized maximized model,Patrickmouth,0,Saint Martin,2016-06-04 09:25:27,0
79.6,39,73392.28,194.23,Organized global flexibility,Lake Deannaborough,0,Rwanda,2016-04-22 07:48:33,0
57.51,38,47682.28,105.71,Re-engineered zero-defect open architecture,Jeffreymouth,0,Moldova,2016-03-31 08:53:43,1
82.3,31,56735.83,232.21,Balanced executive definition,Davieshaven,0,Gabon,2016-04-16 08:36:08,0
73.21,30,51013.37,252.6,Networked logistical info-mediaries,Lake Jessicaville,1,Denmark,2016-05-12 20:57:10,1
79.09,32,69481.85,209.72,Optimized multimedia website,Hernandezchester,1,Svalbard & Jan Mayen Islands,2016-05-07 21:32:51,0
68.47,28,67033.34,226.64,Focused coherent success,North Kennethside,0,Poland,2016-06-25 00:33:23,0
83.69,36,68717.0,192.57,Robust context-sensitive neural-net,Shelbyport,0,Fiji,2016-03-23 05:27:35,0
83.48,31,59340.99,222.72,Intuitive zero administration adapter,Williamport,1,Philippines,2016-03-04 13:47:47,0
43.49,45,47968.32,124.67,Synchronized full-range portal,Smithside,0,Vietnam,2016-06-14 12:08:10,1
66.69,35,48758.92,108.27,Integrated encompassing support,Vanessastad,0,Jersey,2016-05-11 19:13:42,1
48.46,49,61230.03,132.38,Devolved human-resource circuit,Lisamouth,1,Indonesia,2016-01-21 23:33:22,1
42.51,30,54755.71,144.77,Grass-roots transitional flexibility,Lake Rhondaburgh,1,Palestinian Territory,2016-01-15 19:45:33,1
42.83,34,54324.73,132.38,Vision-oriented methodical support,Cunninghamhaven,1,Latvia,2016-04-23 09:42:08,1
41.46,42,52177.4,128.98,Integrated impactful groupware,Robertstown,1,Malta,2016-05-23 08:06:24,1
45.99,33,51163.14,124.61,Face-to-face methodical intranet,South Mark,1,Afghanistan,2016-02-27 15:04:52,1
68.72,27,66861.67,225.97,Fundamental tangible moratorium,New Taylorburgh,0,Austria,2016-02-23 17:37:46,0
63.11,34,63107.88,254.94,Balanced mobile Local Area Network,Port Karenfurt,1,Micronesia,2016-03-17 22:59:46,0
49.21,46,49206.4,115.6,Realigned 24/7 core,Carterland,0,Mexico,2016-02-28 03:34:35,1
55.77,49,55942.04,117.33,Fully-configurable high-level groupware,East Shawn,1,Chile,2016-03-15 14:33:12,1
44.13,40,33601.84,128.48,Ameliorated discrete extranet,West Derekmouth,1,Cuba,2016-03-03 20:20:32,1
57.82,46,48867.36,107.56,Centralized asynchronous portal,Brandiland,1,Belarus,2016-04-06 14:16:52,1
72.46,40,56683.32,113.53,Enhanced tertiary utilization,Cervantesshire,0,Malawi,2016-05-01 09:23:25,1
61.88,45,38260.89,108.18,Balanced disintermediate conglomeration,North Debrashire,0,Afghanistan,2016-05-30 08:02:27,1
78.24,23,54106.21,199.29,Sharable value-added solution,Deannaville,0,Luxembourg,2016-04-04 11:39:51,0
74.61,38,71055.22,231.28,Networked impactful framework,East Christopher,1,South Africa,2016-04-06 23:10:40,0
89.18,37,46403.18,224.01,Public-key impactful neural-net,Rickymouth,1,Nepal,2016-04-26 21:45:50,0
44.16,42,61690.93,133.42,Innovative interactive portal,Port Dennis,1,Spain,2016-05-25 00:34:59,1
55.74,37,26130.93,124.34,Networked asymmetric infrastructure,Lake Michelle,1,Hong Kong,2016-02-11 16:45:41,1
88.82,36,58638.75,169.1,Assimilated discrete strategy,East Johnport,0,Slovakia (Slovak Republic),2016-01-30 00:05:37,0
70.39,32,47357.39,261.52,Phased 5thgeneration open system,Sabrinaview,1,Cayman Islands,2016-07-12 10:56:21,0
59.05,52,50086.17,118.45,Upgradable logistical flexibility,Kristinfurt,1,Uganda,2016-04-23 03:46:34,1
78.58,33,51772.58,250.11,Centralized user-facing service-desk,Chapmanland,1,Vanuatu,2016-04-16 10:36:49,0
35.11,35,47638.3,158.03,Extended analyzing emulation,North Jonathan,1,Anguilla,2016-03-11 13:07:30,1
60.39,45,38987.42,108.25,Front-line methodical utilization,Port Christina,1,Switzerland,2016-03-02 15:39:02,1
81.56,26,51363.16,213.7,Open-source scalable protocol,Juanport,1,Zimbabwe,2016-07-13 21:31:14,0
75.03,34,35764.49,255.57,Networked local secured line,East Mike,0,Uruguay,2016-05-29 18:12:00,1
50.87,24,62939.5,190.41,Programmable empowering orchestration,North Angelatown,0,Liberia,2016-05-10 17:13:47,1
82.8,30,58776.67,223.2,Enhanced systemic benchmark,West Steven,1,Egypt,2016-05-07 08:39:47,0
78.51,25,59106.12,205.71,Focused web-enabled Graphical User Interface,Riggsstad,1,Greece,2016-01-17 13:27:13,0
37.65,51,50457.01,161.29,Automated stable help-desk,Davidview,1,Bahrain,2016-03-09 06:22:03,1
83.17,43,54251.78,244.4,Managed national hardware,Port Kevinborough,1,Sri Lanka,2016-04-05 18:02:49,0
91.37,45,51920.49,182.65,Re-engineered composite moratorium,Lawsonshire,1,Kazakhstan,2016-04-01 07:37:18,1
68.25,29,70324.8,220.08,Phased fault-tolerant definition,Wagnerchester,0,Greenland,2016-02-15 16:18:49,0
81.32,25,52416.18,165.65,Pre-emptive next generation Internet solution,Daisymouth,0,Moldova,2016-03-08 05:12:57,0
76.64,39,66217.31,241.5,Reverse-engineered web-enabled support,North Daniel,1,Poland,2016-02-09 23:38:30,0
74.06,50,60938.73,246.29,Horizontal intermediate monitoring,Port Jacquelinestad,1,Anguilla,2016-06-17 09:38:22,0
39.53,33,40243.82,142.21,Intuitive transitional artificial intelligence,New Teresa,1,Central African Republic,2016-06-01 12:27:17,1
86.58,32,60151.77,195.93,Business-focused asynchronous budgetary management,Henryfort,1,Mexico,2016-02-26 23:44:44,0
90.75,40,45945.88,216.5,Decentralized methodical capability,Lake Joseph,0,Togo,2016-03-11 09:58:32,0
67.71,25,63430.33,225.76,Synergized intangible open system,Daviesborough,1,Armenia,2016-04-28 02:55:10,0
82.41,36,65882.81,222.08,Stand-alone logistical service-desk,North Brandon,0,Nicaragua,2016-04-12 04:22:42,0
45.82,27,64410.8,171.24,Expanded full-range synergy,Adamside,1,Eritrea,2016-02-10 20:43:38,1
76.79,27,55677.12,235.94,Open-architected intangible strategy,Wademouth,0,Canada,2016-05-01 23:21:53,0
70.05,33,75560.65,203.44,Diverse directional hardware,North Raymond,0,Croatia,2016-03-24 17:48:31,0
72.19,32,61067.58,250.32,Balanced discrete approach,Randolphport,1,Switzerland,2016-04-22 19:45:19,0
77.35,34,72330.57,167.26,Total bi-directional success,East Troyhaven,0,Yemen,2016-03-09 12:10:08,0
40.34,29,32549.95,173.75,Object-based motivating instruction set,Clarkborough,0,Tokelau,2016-03-30 05:29:38,1
67.39,44,51257.26,107.19,Realigned intermediate application,Josephberg,0,Armenia,2016-01-24 13:41:38,1
68.68,34,77220.42,187.03,Sharable encompassing database,Lake Jenniferton,1,Equatorial Guinea,2016-07-15 09:42:19,0
81.75,43,52520.75,249.45,Progressive 24/7 definition,Lake Jose,0,Barbados,2016-06-07 05:41:16,0
66.03,22,59422.47,217.37,Pre-emptive next generation strategy,Ashleymouth,0,American Samoa,2016-05-31 23:32:00,0
47.74,33,22456.04,154.93,Open-source 5thgeneration leverage,Henryland,1,Saint Lucia,2016-05-14 14:49:05,1
79.18,31,58443.99,236.96,Open-source holistic productivity,Lake Danielle,0,Algeria,2016-01-10 20:18:21,0
86.81,29,50820.74,199.62,Multi-channeled scalable moratorium,Joshuaburgh,1,Turkmenistan,2016-02-21 16:57:59,0
41.53,42,67575.12,158.81,Optional tangible productivity,South Jeanneport,0,Mayotte,2016-05-23 00:32:54,1
70.92,39,66522.79,249.81,Up-sized intangible circuit,New Nathan,1,South Africa,2016-07-21 20:30:06,0
46.84,45,34903.67,123.22,Virtual homogeneous budgetary management,Jonesshire,0,Macao,2016-05-15 18:44:50,1
44.4,53,43073.78,140.95,Phased zero-defect portal,Mariahview,1,France,2016-06-30 00:43:40,1
52.17,44,57594.7,115.37,Optional modular throughput,New Julianberg,1,Equatorial Guinea,2016-02-24 06:17:18,1
81.45,31,66027.31,205.84,Triple-buffered human-resource complexity,Randyshire,1,Mali,2016-05-30 21:22:22,0
54.08,36,53012.94,111.02,Innovative cohesive pricing structure,Philipberg,1,Mayotte,2016-06-02 04:14:37,1
76.65,31,61117.5,238.43,Function-based executive moderator,West Dennis,0,Pakistan,2016-04-18 07:00:38,0
54.39,20,52563.22,171.9,Digitized content-based circuit,Richardshire,1,Guadeloupe,2016-02-29 18:06:21,1
37.74,40,65773.49,190.95,Balanced uniform algorithm,Lake James,0,Denmark,2016-05-27 12:45:37,1
69.86,25,50506.44,241.36,Triple-buffered foreground encryption,Austinborough,0,New Zealand,2016-01-12 21:17:15,0
85.37,36,66262.59,194.56,Front-line system-worthy flexibility,Alexandrafort,1,Netherlands Antilles,2016-01-27 17:08:19,0
80.99,26,35521.88,207.53,Centralized clear-thinking Graphic Interface,Melissastad,1,Belarus,2016-06-10 03:56:41,0
78.84,32,62430.55,235.29,Optimized 5thgeneration moratorium,Gonzalezburgh,1,Taiwan,2016-04-09 09:26:39,0
77.36,41,49597.08,115.79,Fully-configurable asynchronous firmware,Port Jennifer,0,El Salvador,2016-02-26 06:00:16,1
55.46,37,42078.89,108.1,Exclusive systematic algorithm,Chrismouth,0,Taiwan,2016-02-21 23:07:11,1
35.66,45,46197.59,151.72,Exclusive cohesive intranet,Port Beth,0,Peru,2016-04-29 14:08:26,1
50.78,51,49957.0,122.04,Vision-oriented asynchronous Internet solution,West David,0,Liberia,2016-02-11 17:02:07,1
40.47,38,24078.93,203.9,Sharable 5thgeneration access,Fraziershire,0,Burundi,2016-07-22 07:44:43,1
45.62,43,53647.81,121.28,Monitored homogeneous artificial intelligence,Robertfurt,0,Macao,2016-06-26 02:34:15,1
84.76,30,61039.13,178.69,Monitored 24/7 moratorium,South Pamela,0,Venezuela,2016-05-14 23:08:14,0
80.64,26,46974.15,221.59,Vision-oriented real-time framework,North Laurenview,0,Luxembourg,2016-05-24 10:04:39,0
75.94,27,53042.51,236.96,Future-proofed stable function,Campbellstad,1,Italy,2016-02-16 12:05:45,0
37.01,50,48826.14,216.01,Secured encompassing Graphical User Interface,Port Derekberg,0,San Marino,2016-03-20 02:44:13,1
87.18,31,58287.86,193.6,Right-sized logistical middleware,West Andrew,0,Madagascar,2016-01-31 05:12:44,0
56.91,50,21773.22,146.44,Team-oriented executive core,West Randy,0,Norfolk Island,2016-04-01 05:17:28,1
75.24,24,52252.91,226.49,Vision-oriented next generation solution,South Christopher,0,Vanuatu,2016-02-25 16:33:24,0
42.84,52,27073.27,182.2,Enhanced optimizing website,Lake Michellebury,1,Tunisia,2016-03-21 11:02:49,1
67.56,47,50628.31,109.98,Reduced background data-warehouse,Zacharyton,0,Paraguay,2016-02-12 05:20:19,1
34.96,42,36913.51,160.49,Right-sized mobile initiative,West James,1,Macedonia,2016-06-01 16:10:30,1
87.46,37,61009.1,211.56,Synergized grid-enabled framework,Millerview,1,Heard Island and McDonald Islands,2016-06-16 03:17:45,0
41.86,39,53041.77,128.62,Open-source stable paradigm,Hawkinsbury,1,Ethiopia,2016-03-26 15:28:07,1
34.04,34,40182.84,174.88,Reverse-engineered context-sensitive emulation,Elizabethport,1,El Salvador,2016-02-16 07:37:28,1
54.96,42,59419.78,113.75,Public-key disintermediate emulation,West Amanda,1,Niger,2016-02-28 09:31:31,1
87.14,31,58235.21,199.4,Up-sized bifurcated capability,Wadestad,1,Timor-Leste,2016-05-18 01:00:52,0
78.79,32,68324.48,215.29,Stand-alone background open system,Mauriceshire,1,Uruguay,2016-02-21 13:11:08,0
65.56,25,69646.35,181.25,Stand-alone explicit orchestration,West Arielstad,1,Somalia,2016-01-05 12:59:07,0
81.05,34,54045.39,245.5,Configurable asynchronous application,Adamsstad,0,Malaysia,2016-05-18 00:07:43,0
55.71,37,57806.03,112.52,Upgradable 4thgeneration portal,Lake James,1,Korea,2016-03-06 23:26:44,1
45.48,49,53336.76,129.16,Networked client-server solution,Blairborough,1,Lao People's Democratic Republic,2016-05-19 04:23:41,1
47.0,56,50491.45,149.53,Public-key bi-directional Graphical User Interface,New Marcusbury,0,Bahamas,2016-04-29 20:40:21,1
59.64,51,71455.62,153.12,Re-contextualized human-resource success,Evansville,1,Guyana,2016-05-03 01:09:01,1
35.98,45,43241.88,150.79,Front-line fresh-thinking installation,Huffmanchester,0,Ethiopia,2016-06-27 21:51:47,1
72.55,22,58953.01,202.34,Balanced empowering success,New Cynthia,0,Bosnia and Herzegovina,2016-02-08 07:33:22,0
91.15,38,36834.04,184.98,Robust uniform framework,Joshuamouth,0,Cyprus,2016-02-22 07:04:05,0
80.53,29,66345.1,187.64,Sharable upward-trending support,West Benjamin,0,Singapore,2016-03-21 08:13:24,0
82.49,45,38645.4,130.84,Assimilated multi-state paradigm,Williamsfort,0,Dominican Republic,2016-05-31 00:58:37,1
80.94,36,60803.0,239.94,Self-enabling local strategy,North Tiffany,0,Bermuda,2016-01-01 05:31:22,0
61.76,34,33553.9,114.69,Open-source local approach,Edwardsport,0,Jamaica,2016-05-27 08:53:51,1
63.3,38,63071.34,116.19,Polarized intangible encoding,Lake Evantown,0,Saint Barthelemy,2016-05-09 07:13:27,1
36.73,34,46737.34,149.79,Multi-lateral attitude-oriented adapter,South Henry,1,Albania,2016-06-27 01:56:36,1
78.41,33,55368.67,248.23,Multi-lateral 24/7 Internet solution,Harmonhaven,1,Mozambique,2016-06-03 04:51:46,0
83.98,36,68305.91,194.62,Profit-focused secondary portal,West Gregburgh,0,Zimbabwe,2016-02-24 00:44:44,0
63.18,45,39211.49,107.92,Reactive upward-trending migration,Hansenland,0,Georgia,2016-03-05 12:03:41,1
50.6,48,65956.71,135.67,Customer-focused fault-tolerant implementation,Port Michaelmouth,0,Brazil,2016-01-15 22:49:45,1
32.6,38,40159.2,190.05,Customizable homogeneous contingency,Tylerport,0,Syrian Arab Republic,2016-02-12 03:39:09,1
60.83,19,40478.83,185.46,Versatile next generation pricing structure,West Lacey,1,Palestinian Territory,2016-02-19 20:49:27,0
44.72,46,40468.53,123.86,Cross-group systemic customer loyalty,North Jenniferburgh,1,Grenada,2016-03-12 02:48:18,1
78.76,51,66980.27,162.05,Face-to-face modular budgetary management,South Davidhaven,0,Ghana,2016-07-23 04:04:42,1
79.51,39,34942.26,125.11,Proactive non-volatile encryption,North Charlesbury,1,Brunei Darussalam,2016-03-06 09:33:46,1
39.3,32,48335.2,145.73,Decentralized bottom-line help-desk,Jonathanland,0,Lithuania,2016-02-24 04:11:37,1
64.79,30,42251.59,116.07,Visionary mission-critical application,North Virginia,0,Maldives,2016-02-17 20:22:49,1
89.8,36,57330.43,198.24,User-centric attitude-oriented adapter,West Tanner,0,Lesotho,2016-02-02 04:57:50,0
72.82,34,75769.82,191.82,User-centric discrete success,Jonesmouth,1,Czech Republic,2016-01-27 16:06:05,0
38.65,31,51812.71,154.77,Total even-keeled architecture,Port Jason,1,Iceland,2016-05-24 09:50:41,1
59.01,30,75265.96,178.75,Focused multimedia implementation,West Annefort,1,Philippines,2016-02-08 22:45:26,1
78.96,50,69868.48,193.15,Stand-alone well-modulated product,East Jason,0,Cayman Islands,2016-02-12 01:55:38,1
63.99,43,72802.42,138.46,Ameliorated bandwidth-monitored contingency,North Cassie,0,Haiti,2016-01-11 08:18:12,1
41.35,27,39193.45,162.46,Streamlined homogeneous analyzer,Hintonport,1,Colombia,2016-03-03 03:51:27,1
62.79,36,18368.57,231.87,Total coherent archive,New James,1,Luxembourg,2016-05-30 20:08:51,1
45.53,29,56129.89,141.58,Front-line neutral alliance,North Destiny,0,United Arab Emirates,2016-04-22 22:01:21,1
51.65,31,58996.56,249.99,Virtual context-sensitive support,Mclaughlinbury,0,Ireland,2016-05-25 10:39:28,0
54.55,44,41547.62,109.04,Re-engineered optimal policy,West Gabriellamouth,0,Canada,2016-02-04 03:10:17,1
35.66,36,59240.24,172.57,Implemented uniform synergy,Alvarezland,0,Svalbard & Jan Mayen Islands,2016-02-21 20:09:12,1
69.95,28,56725.47,247.01,Horizontal even-keeled challenge,New Julie,0,Malta,2016-04-28 01:24:34,0
79.83,29,55764.43,234.23,Innovative regional groupware,North Frankstad,1,Sudan,2016-05-18 19:33:51,0
85.35,37,64235.51,161.42,Exclusive multi-state Internet solution,Claytonside,1,Ecuador,2016-02-17 11:15:31,0
56.78,28,39939.39,124.32,Mandatory empowering focus group,Melanieton,0,Senegal,2016-06-19 23:04:45,1
78.67,26,63319.99,195.56,Proactive 5thgeneration frame,Lake Michaelport,0,Cambodia,2016-02-20 09:54:06,0
70.09,21,54725.87,211.17,Automated full-range Internet solution,East Benjaminville,0,Belarus,2016-01-22 12:58:14,0
60.75,42,69775.75,247.05,Fully-configurable systemic productivity,Garrettborough,1,Guyana,2016-02-19 13:26:24,0
65.07,24,57545.56,233.85,Multi-lateral multi-state encryption,Port Raymondfort,0,Mali,2016-01-03 07:13:53,0
35.25,50,47051.02,194.44,Intuitive global website,Waltertown,0,Iran,2016-01-03 04:39:47,1
37.58,52,51600.47,176.7,Exclusive disintermediate Internet solution,Cameronberg,1,Bulgaria,2016-04-13 13:04:47,1
68.01,25,68357.96,188.32,Ameliorated actuating workforce,Kaylashire,1,Afghanistan,2016-01-01 03:35:35,0
45.08,38,35349.26,125.27,Synergized clear-thinking protocol,Fosterside,0,Liberia,2016-03-27 08:32:37,1
63.04,27,69784.85,159.05,Triple-buffered multi-state complexity,Davidstad,0,Netherlands Antilles,2016-07-10 16:25:56,1
40.18,29,50760.23,151.96,Enhanced intangible portal,Lake Tracy,0,Hong Kong,2016-06-25 04:21:33,1
45.17,48,34418.09,132.07,Down-sized background groupware,Taylormouth,1,Palau,2016-01-27 14:41:10,1
50.48,50,20592.99,162.43,Switchable real-time product,Dianaville,0,Malawi,2016-05-16 18:51:59,1
80.87,28,63528.8,203.3,Ameliorated local workforce,Collinsburgh,0,Uruguay,2016-02-27 20:20:25,0
41.88,40,44217.68,126.11,Streamlined exuding adapter,Port Rachel,1,Cyprus,2016-02-28 23:54:44,1
39.87,48,47929.83,139.34,Business-focused user-facing benchmark,South Rebecca,1,Mexico,2016-06-13 06:11:33,1
61.84,45,46024.29,105.63,Reactive bi-directional standardization,Port Joshuafort,1,Niger,2016-05-05 11:07:13,1
54.97,31,51900.03,116.38,Virtual bifurcated portal,Robinsontown,1,France,2016-07-07 12:17:33,1
71.4,30,72188.9,166.31,Integrated 3rdgeneration monitoring,Beckton,0,Japan,2016-05-24 17:07:08,0
70.29,31,56974.51,254.65,Balanced responsive open system,New Frankshire,1,Norfolk Island,2016-03-30 14:36:55,0
67.26,57,25682.65,168.41,Focused incremental Graphic Interface,North Derekville,1,Bulgaria,2016-05-27 05:54:03,1
76.58,46,41884.64,258.26,Secured 24hour policy,West Sydney,0,Uzbekistan,2016-01-03 16:30:51,0
54.37,38,72196.29,140.77,Up-sized asymmetric firmware,Lake Matthew,0,Mexico,2016-06-25 18:17:53,1
82.79,32,54429.17,234.81,Distributed fault-tolerant service-desk,Lake Zacharyfurt,1,Brunei Darussalam,2016-02-24 10:36:43,0
66.47,31,58037.66,256.39,Vision-oriented human-resource synergy,Lindsaymouth,1,France,2016-03-03 03:13:48,0
72.88,44,64011.26,125.12,Customer-focused explicit challenge,Sarahland,0,Yemen,2016-04-21 19:56:24,1
76.44,28,59967.19,232.68,Synchronized human-resource moderator,Port Julie,1,Northern Mariana Islands,2016-04-06 17:26:37,0
63.37,43,43155.19,105.04,Open-architected full-range projection,Michaelshire,1,Poland,2016-03-23 12:53:23,1
89.71,48,51501.38,204.4,Versatile local forecast,Sarafurt,1,Bahrain,2016-02-17 07:00:38,0
70.96,31,55187.85,256.4,Ameliorated user-facing help-desk,South Denise,0,Saint Pierre and Miquelon,2016-06-26 07:01:47,0
35.79,44,33813.08,165.62,Enterprise-wide tangible model,North Katie,1,Tonga,2016-04-20 13:36:42,1
38.96,38,36497.22,140.67,Versatile mission-critical application,Mauricefurt,1,Comoros,2016-07-21 16:02:40,1
69.17,40,66193.81,123.62,Extended leadingedge solution,New Patrick,0,Montenegro,2016-03-06 11:36:06,1
64.2,27,66200.96,227.63,Phased zero tolerance extranet,Edwardsmouth,1,Isle of Man,2016-02-11 23:45:01,0
43.7,28,63126.96,173.01,Front-line bifurcated ability,Nicholasland,0,Mayotte,2016-04-04 03:57:48,1
72.97,30,71384.57,208.58,Fundamental modular algorithm,Duffystad,1,Lebanon,2016-02-11 21:49:00,1
51.3,45,67782.17,134.42,Grass-roots cohesive monitoring,New Darlene,1,Bosnia and Herzegovina,2016-04-22 02:07:01,1
51.63,51,42415.72,120.37,Expanded intangible solution,South Jessica,1,Mongolia,2016-02-01 17:24:57,1
55.55,19,41920.79,187.95,Proactive bandwidth-monitored policy,West Steven,0,Guatemala,2016-03-24 02:35:54,0
45.01,26,29875.8,178.35,Virtual 5thgeneration emulation,Ronniemouth,0,Brazil,2016-06-03 21:43:21,1


