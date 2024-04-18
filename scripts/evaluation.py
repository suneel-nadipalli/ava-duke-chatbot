import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from api_inference import query

from datasets import load_dataset

import nltk

from nltk.tokenize import word_tokenize

nltk.download('punkt')

import sys

sys.path.append('..')

def stringent_acc():
    """
    Purpose: Function to evaluate the stringent factual accuracy of the model
    """
    #Stringent Factual Accuracy--> cosine similarity is not necessarily useful; questions with one specific number answer, specific yes or no, etc, rather than open ended
    #both cosine similarity to actual answer and human evaluation

    # Set up evaluation questions

    q_1 = "Which specific courses are included in the Fall 2021 schedule for AIPI students at Pratt Institute?"
    context_1 = "Academics 1. What classes are being offered to AIPI students in Fall 2021? In the Fall semester of the AIPI program students take a fixed schedule of courses (electives are taken in the Spring). Students should plan to register for the following courses: - AIPI 503: Bootcamp [0 units] (On-campus, Online MEng, Online Certificate students) - AIPI 510: Sourcing Data for Analytics [3 units] (On-campus, Online MEng, Online Certificate students) - AIPI 520: Modeling Process & Algorithms [3 units] (On-campus & Online MEng students) - AIPI 530: AI in Practice [3 units] (On-campus students) - MENG 570: Business Fundamentals for Engineers [3 units] (On-campus students completing in 12 months) - AIPI 501: Industry Seminar Series [0 units] (On-campus & Online MEng students) The full list of Pratt courses will be made available to students on June 28, 2021, when shopping carts open in DukeHub. 2. When will the list of Fall 2021 courses be available? The list of all Fall 2021 courses offered by the Pratt School of Engineering will be made available to all incoming and continuing students on June 28, 2021, the day that shopping carts open. This list can be accessed using the Simple and Advanced Class Search functions in DukeHub. 3. When can I register for classes? Fall 2021 course registration for all graduate students will begin on July 7, 2021, last day that I can register for Fall 2021 classes? Students may make changes to their schedule up to the end of the Drop/Add period, which ends on September 3, 2021. 12. For Campus students, is there a formal process for selecting between the 12-month and 16-month plan? No, there is no specific documentation that must be provided if you choose to extend the AIPI Program to three semesters. However, the earlier you make the decision the better as it will likely affect your selection of courses in Fall and Spring semesters. 13. When do classes start? Fall 2021 classes begin on August 23, 2021. 14. What are the class meeting patterns for AIPI courses? Each course typically meets once per week for 2 hours and 45 minutes. 15. How much time is required outside of class for AIPI courses? AIPI classes are challenging and require significant work outside of class. Each week, you should expect to spend between 8-10 hours per course working outside of class. 16. How will classes be offered in Fall 2021 (e.g., all in-person, online, or a mix of the two)? While Duke will offer classes fully in-person for the Fall 2021 semester, we understand that the COVID-19 pandemic has created travel and visa challenges for many of our international students. AIPI classes will be offered both in-person and online for the Fall 2021 semester. Students who will be physically located in Durham, NC, USA, for the semester are expected to come to class in-person incoming and continuing students on June 28, 2021, the day that shopping carts open. This list can be accessed using the Simple and Advanced Class Search functions in DukeHub. 3. When can I register for classes? Fall 2021 course registration for all graduate students will begin on July 7, 2021, in DukeHub. Students may continue making changes to their schedule during the Drop/Add period, which will end on September 3. After Drop/Add ends, students may no longer change their Fall 2021 schedules in DukeHub. 4. How do I register for classes? All students register for classes through DukeHub, the self-service application that provides students with an array of information and direct access to academic, financial, and personal data. Access to DukeHub is gained using your NetID and password. For assistance with registration, please see the help guides linked here. 5. What classes outside of the AIPI curriculum can I count toward my technical electives requirement? Approved AIPI electives are listed on the AIPI website. Additionally, students may take technical electives in other programs and departments across campus with approval from both the course instructor and the AIPI Director. Please note that graduate courses outside of AIPI require an instructor’s permission to enroll, and it is ultimately at the instructor’s discretion to determine whether or not AIPI students will be allowed into a particular class. 6. Can I audit classes? Yes, AIPI students may audit courses on a space-available basis with consent of the instructor and the AIPI Director. Audited courses. Begin taking these classes online at the beginning of the semester, and if you are able to reach the United States by September 22, you will be able to continue your studies in-person. If you are unable to reach the United States by September 22, you can continue taking your classes online for the Fall 2021 semester. 35. Whom do I contact about other visa questions I may have? In the Duke Visa Services Office, Betty Henderson (betty.henderson@duke.edu) is the Visa Services Advisor for Pratt students. You may also contact Kelsey Liddle (kelsey.liddle@duke.edu), the Pratt Student Records Coordinator, for additional visa-related questions. Miscellaneous 36. How do I get my NetID and password? You should receive a separate email from the Office of Information Technology (OIT) with instructions to set up your NetID and email alias. Your NetID is your electronic key to online resources, including your Duke email account, DukeHub, Sakai, MyDuke, Box cloud storage, and more. Please set up your NetID as soon as possible. 37. When will I get access to my Duke email? Once you set up your NetID (see above), you will be able to access your email. This site provides information about how to access your Duke email account. Once you have access to your Duke email, please begin checking and using this address. Your Duke email address will become the primary source for all your communications from Duke. 38. What can I do this summer to prepare for AIPI? While some of our students able to participate in classes and make progress toward their degrees. In the AIPI Program, if you anticipate that you will be able to make it to Durham in time to participate in-person (i.e., by September 22), you should register for the hybrid sections of the courses that you enroll in. You may also enroll in the fully-online section of a course if you know that you will be unable to travel to the United States for the Fall 2021 semester. 34. I hope to arrive in the United States for Fall 2021 classes, but it might be after drop/add ends. What should I do? Students on F-1 visas have until 30 days after the first day of class to arrive in the United States. For Fall 2021, classes begin on August 23, so students must arrive in the United States by September 22. Drop/Add ends on September 3, after which students may not make changes to their Fall 2021 course schedules. Therefore, we recommend that if you anticipate arriving in the United States between September 3 and September 22, you enroll in Fall classes that are being offered according to the Hybrid (i.e., both in-person and online accessible) model. Begin taking these classes online at the beginning of the semester, and if you are able to reach the United States by September 22, you will be able to continue your studies in-person. If you are unable to reach the United States by September 22, you can continue taking your"
    real_answer_1 = "The Fall 2021 schedule for AIPI students at Pratt Institute includes the following courses: 1. AIPI 503: Bootcamp [0 units] (On-campus, Online MEng, Online Certificate students) 2. AIPI 510: Sourcing Data for Analytics [3 units] (On-campus, Online MEng, Online Certificate students) 3. AIPI 520: Modeling Process & Algorithms [3 units] (On-campus & Online MEng students) 4. AIPI 530: AI in Practice [3 units] (On-campus students) 5. MENG 570: Business Fundamentals for Engineers [3 units] (On-campus students completing in 12 months) 6. AIPI 501: Industry Seminar Series [0 units] (On-campus & Online MEng students) Please note that this information is based on the provided details regarding the Fall 2021 course offerings for AIPI students at Pratt Institute."


    q_2 = "Are students at Duke University's AIPI program required to stick to one elective track, or can they freely select courses aligned with their professional goals?"
    context_2 = "last day that I can register for Fall 2021 classes? Students may make changes to their schedule up to the end of the Drop/Add period, which ends on September 3, 2021. 12. For Campus students, is there a formal process for selecting between the 12-month and 16-month plan? No, there is no specific documentation that must be provided if you choose to extend the AIPI Program to three semesters. However, the earlier you make the decision the better as it will likely affect your selection of courses in Fall and Spring semesters. 13. When do classes start? Fall 2021 classes begin on August 23, 2021. 14. What are the class meeting patterns for AIPI courses? Each course typically meets once per week for 2 hours and 45 minutes. 15. How much time is required outside of class for AIPI courses? AIPI classes are challenging and require significant work outside of class. Each week, you should expect to spend between 8-10 hours per course working outside of class. 16. How will classes be offered in Fall 2021 (e.g., all in-person, online, or a mix of the two)? While Duke will offer classes fully in-person for the Fall 2021 semester, we understand that the COVID-19 pandemic has created travel and visa challenges for many of our international students. AIPI classes will be offered both in-person and online for the Fall 2021 semester. Students who will be physically located in Durham, NC, USA, for the semester are expected to come to class in-person incoming and continuing students on June 28, 2021, the day that shopping carts open. This list can be accessed using the Simple and Advanced Class Search functions in DukeHub. 3. When can I register for classes? Fall 2021 course registration for all graduate students will begin on July 7, 2021, in DukeHub. Students may continue making changes to their schedule during the Drop/Add period, which will end on September 3. After Drop/Add ends, students may no longer change their Fall 2021 schedules in DukeHub. 4. How do I register for classes? All students register for classes through DukeHub, the self-service application that provides students with an array of information and direct access to academic, financial, and personal data. Access to DukeHub is gained using your NetID and password. For assistance with registration, please see the help guides linked here. 5. What classes outside of the AIPI curriculum can I count toward my technical electives requirement? Approved AIPI electives are listed on the AIPI website. Additionally, students may take technical electives in other programs and departments across campus with approval from both the course instructor and the AIPI Director. Please note that graduate courses outside of AIPI require an instructor’s permission to enroll, and it is ultimately at the instructor’s discretion to determine whether or not AIPI students will be allowed into a particular class. 6. Can I audit classes? Yes, AIPI students may audit courses on a space-available basis with consent of the instructor and the AIPI Director. Audited courses track? No, there is not currently a formal process to designate your elective track. We do not require students to rigidly adhere to one elective track. Students may choose electives that fit their professional goals. The elective tracks are meant as guides for students to align and develop skills toward a particular area, and those students who complete a track may list it on their resume. 20. What do I do if I want to change my elective track? If you wish to change your elective track, there is no formal action that you need to take. However, it is a good idea to speak with the program director about your elective course plans, as they can help steer you toward courses that align with your professional aspirations. 21. How can I track my degree progress? Students can track their degree progress using Stellic, a self-service tool that enables students to see which classes they have taken toward their degree and plan for future semesters. Students are strongly encouraged to use Stellic throughout the course of the AIPI Program so that they can stay on track to graduate within the timeframe they choose (two or three semesters). Tuition and Billing 22. How much does it cost to audit a course? For AIPI students who pay tuition on a pay-by-semester basis (as is the case for all full-time residential AIPI students), there is no charge for auditing a course. For AIPI Online students who pay tuition on a pay-by-credit basis, there Campus students take a fixed set of 4 courses in Fall semester (AIPI 510,520,530 and MENG 570). AIPI Online students typically take 2 courses per semester (Fall: AIPI 510,520) and AIPI Certificate students take 1 course/semester (Fall: AIPI 510). Online students pay tuition on a per-credit basis rather than a flat-rate per-semester basis. 9. What is the limit on credits I can take each semester? AIPI students may take up to 15.0 credits per semester. Full-time residential students on the pay-by-semester basis may take a fifth credit for free each semester (although we generally suggest a maximum of 4 courses as the workload can be intense). Students who attempt to enroll in more than 15.0 credits per semester will not be able to register. 10. What is Drop/Add? What happens during the Drop/Add period? The Drop/Add period occurs after the initial Registration window and continues until the end of the second week of classes. During the Drop/Add period, students may make changes to their schedules through DukeHub. At the end of the Drop/Add period (September 3, 2021), students’ schedules may no longer be changed in DukeHub and can only be changed with permission from their dean. 11. What is the last day that I can register for Fall 2021 classes? Students may make changes to their schedule up to the end of the Drop/Add period, which ends on September 3, 2021. 12. For Campus students, is there a formal process for selecting between the 12-month and 16-month plan? No, there find out about student activities and clubs to participate in? There is a wide variety of student activities and clubs to get involved in at Duke. A partial list of activities/clubs of interest is contained on the AIPI student website. You are also encouraged to check out the Duke Event Calendar to check out upcoming events in a variety of topics, including the arts, athletics, academics, and civic engagement. For International Students 33. I am an international student hoping to come to the United States for Fall 2021, but my home country is backlogged with visa applications due to COVID-19. What should I do? You have several options to choose from in this situation: (a) defer your enrollment for up to one year; (b) enroll in courses with a hybrid format if you believe you can arrive in the United States within 30 days of the first day of class; or (c) enroll in all online courses, remain in your home country for the Fall 2021 semester, and join us on campus in Spring 2022. Duke will be offering Fall 2021 courses in a hybrid format, so that students who are unable to travel to the United States are still able to participate in classes and make progress toward their degrees. In the AIPI Program, if you anticipate that you will be able to make it to Durham in time to participate in-person (i.e., by September 22), you should register for the hybrid sections of the courses that you enroll"
    real_answer_2 = "Students in Duke University's AIPI program are not required to stick to one elective track. They have the freedom to select courses that align with their professional goals. The elective tracks are meant to guide students in developing skills in a particular area, but students are not rigidly bound to one track. They can choose electives that best suit their aspirations and may even change their elective track without any formal action required."

    q_3 = "What is the total cost of for this program, including tuition and additional fees?"
    context_3 = "over two semesters and a summer session would result in a total tuition cost of $75,877. The internship course does not incur tuition charges. Fall 2024: - Tuition: $32,990 - Health Fee: $487 - Health Insurance: $3,381 - Graduate Student Activity Fee: $18 - Graduate Student Service Fee: $12 - Transcript Fee: $120 - Recreation Fee: $190 - Room: $6,008 - Board: $1880 - Book & Supplies: $322 - Local Transportation: $904 - Personal & Misc.: $1,896 - Total Cost of Attendee for Fall 2024: $48,208 Spring 2025: - Tuition: $32,990 - Health Fee: $487 - Health Insurance: $0 - Graduate Student Activity Fee: $18 - Graduate Student Service Fee: $12 - Transcript Fee: $0 - Recreation Fee: $190 - Room: $7,510 - Board: $2,350 - Book & Supplies: $322 - Local Transportation: $1,130 - Personal & Misc.: $2,370 - Total Cost of Attendee for Spring 2025: $47,379 Summer 2025: - Tuition: $9,897 - Health Fee: $225 - Health Insurance: - Graduate Student Activity Fee: - Graduate Student Service Fee: - Transcript Fee: - Recreation Fee: - Room: $4,506 - Board: $1,410 - Book & Supplies: $167 - Local Transportation: $678 - Personal & Misc.: $1,422 - Total Cost of Attendee for Summer 2025: $18,305 *Tuition, fees, and estimates are subject to confirmation each May DURATION The normal duration of the Master of Engineering in AI program is one year of study (2 semesters and a summer session); however, the program can be extended for an additional Fall semester to Attendee for Summer 2025: $18,305 *Tuition, fees, and estimates are subject to confirmation each May DURATION The normal duration of the Master of Engineering in AI program is one year of study (2 semesters and a summer session); however, the program can be extended for an additional Fall semester to complete in 16 months. *The estimated tuition cost of the Extended Track is $85,774 - $95,671, depending on number of credits taken in the final semester. The normal load is four courses (12 units) per semester in the first year Tuition, fees, and expense estimates are subject to confirmation each May Purchase of health insurance is required unless you can show proof of comparable private insurance coverage Online Master's Program Tuition for online Duke Master of Engineering programs for the 2024-2025 academic year is $9,897 per course taken at the university. In general, completion of the 30 required credits over five semesters would result in a total tuition cost of $98,970. Please note that the internship courses do not incur tuition charges. 2024-2025 ONLINE TUITION, FEES, AND ESTIMATED EXPENSES - Tuition: $98,970 - Transcript Fee: $120 - Books: $644 - Total: $99,734 Notes * Students typically take two courses per semester. Tuition, fees, and estimates are subject to confirmation each May. Rates subject to change Also: Domestic students can estimate $438 in loan fees per semester if securing student loans. No tuition is charged for course credits received for the internship, internship assessment, or residency courses. There is a room Online(part-time): - Time to Degree: 24 months - Python & Data Science Math Boot Camp: online 4-week part-time - Class Experience: live and recorded classes - Class Experience: online interaction with peers and faculty - Professional Development: two spring residences on campus at Duke - Professional Development: industry seminar series - Academic Advising: online interaction with a faculty advisor - Academic Advising: in-person interaction during on-campus residencies - Career Services & Professional Development: support from career services professionals specialized in assisting engineering master's students On-campus (full-time): - Time to Degree: 12 months or 16 months - Python & Data Science Math Boot Camp: Online 4-week part-time - Class Experience: Class attendance at Duke - Class Experience: In-person and online interaction with faculty and peers - Professional Development: Industry seminar series - Academic Advising: In-person and online interaction with a faculty advisor - Career Services & Professional Development: Support from career services professionals specialized in assisting engineering master's students 6-week Career Strategy and Design workshop 2024-2025 CAMPUS TUITION, FEES, AND ESTIMATED EXPENSES Tuition for campus-based Duke Master of Engineering programs for the 2024-2025 academic year is $32,990 per semester taken at the university. Tuition for the Master of Engineering in AI over two semesters and a summer session would result in a total tuition cost of $75,877. The internship course does not incur tuition charges. Fall 2024: - Tuition: $32,990 - Health Fee: $487 - Health Insurance: $3,381 - Graduate Student Activity Fee: $18 - Graduate Student Service Fee: $12 - assuming normal time to completion. There may be additional costs for living expenses. Note that this information is subject to change: - Tuition: $8,364 per 3-credit class, and a total of $33,456 for the certificate ** - Transcript Fee: $120 - Books: $322 - Total: $33,898 Notes ** Duke may change its tuition for each academic year, and this estimate is based on current academic year tuition charges Student Loans Information about average student debt, monthly debt expense and debt interest rate have been withheld since fewer than 10 students have completed this program Job Placement Rates We are not currently required to calculate a job placement rate for program completers Program graduates are employed in the following fields: Information Technology Manufacturing Science, Technology, Engineering, and Mathematics Transportation, Distribution, and Logistics The program does not meet any licensing requirements Additional Information—Date Created: 3/15/2021 * These disclosures are required by the U.S. Department of Education For more information, please visit meng.duke.edu » Limited merit-based financial aid is available to highly qualified candidates through academic scholarships emphasizing increasing diversity within the program. U.S. Citizens or Permanent Residents who are underrepresented minorities may receive up to 50 percent per year in tuition scholarships through our Diversity Scholarships. All applicants to the AI program are considered for available financial assistance at the time of program application. More information is available at meng.duke.edu » The Pratt School of Engineering's 4+1: BSE+Master option allows Duke students to earn an undergraduate degree and a master's in five for US visa sponsorship. Application Fee US$75 Paid by credit card with your application. Fee waivers » Documentation of your Bachelor's Degree, in engineering or science from an accredited institution: Transcripts (or, for institutions not using a 4.0 system, estimated GPA and grade scale) Other Items: Short Answer Essays Resume Three (3) Recommendations Video Introduction International Applicants: English Language Testing official results Optional for 2024 Applicants: Graduate Record Exam (GRE) official results, or equivalent exam For Fall Entry The Artificial Intelligence Master of Engineering and Graduate Certificate programs review applications on a rolling basis. Applications submitted earlier than the deadlines listed below will likely receive an earlier response. ON-CAMPUS Master of Engineering Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by March 15; Decision Notification by April 15; Reply Required by May 1. ONLINE Master of Engineering Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by April 15; Decision Notification by May 15; Reply Required by June 1. ONLINE Certificate Program Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by April 15; Decision Notification by May 15; Reply Required by June 1."
    real_answer_3 = "The total cost of pursuing the Duke Master of Engineering in Artificial Intelligence program can vary depending on whether you choose the on-campus full-time option, online part-time option, or the extended track option. Here are the estimated total costs for each: 1. On-Campus Full-Time Option (2 semesters and a summer session): - Fall 2024: $48,208 - Spring 2025: $47,379 - Summer 2025: $18,305 Total: $113,892 ,2. Online Part-Time Option (5 semesters): - Total tuition cost: $98,970 - Additional fees and expenses: $764 Total: $99,734 ,3. Extended Track Option (16 months): - Estimated tuition cost: $85,774 - $95,671 (depending on number of credits taken in the final semester) - Additional fees and expenses: varies Total: varies"

    q_4 = "Are the deadlines for the online Certificate Program similar to those for the on-campus Master of Engineering program?"
    context_4 = "Duke AI for Product Innovation Master of Engineering program will additionally be required to provide GRE scores (if required at time of application). GRE scores are optional for the 2023-24 admissions cycle. Applicants who are accepted will then need to complete the remaining requirements for the degree: typically, four electives, two Master of Engineering management core courses, and the required on-campus residencies in Durham, NC. This is not an offer of preferential admission, and there is no guarantee of admission. See our academic policy bulletin for the most current details. This standalone online certificate program is not yet eligible for VA benefits or federal student aid. Please check back for updates. Please Note: This standalone online certificate program does NOT qualify students for U.S. visa sponsorship. PROGRAM LENGTH Typically 15 months (1 course per semester, including summer) STUDENT LOANS Information about average student debt, monthly debt expense and debt interest rate have been withheld since fewer than 10 students have completed this program JOB PLACEMENT RATES We are not currently required to calculate a job placement rate for program completers Program graduates are employed in the following fields: Information Technology Manufacturing Science, Technology, Engineering, and Mathematics Transportation, Distribution, and Logistics The program does not meet any licensing requirements Additional Information: Date Created 3/15/2021 * These disclosures are required by the U.S. Department of Education Faculty Director - Jon Reifschneider AI AND MACHINE LEARNING TECHNICAL FACULTY - Brinnae Bent: Adjunct Assistant Professor in the Engineering Graduate and Professional Programs - Xu for US visa sponsorship. Application Fee US$75 Paid by credit card with your application. Fee waivers » Documentation of your Bachelor's Degree, in engineering or science from an accredited institution: Transcripts (or, for institutions not using a 4.0 system, estimated GPA and grade scale) Other Items: Short Answer Essays Resume Three (3) Recommendations Video Introduction International Applicants: English Language Testing official results Optional for 2024 Applicants: Graduate Record Exam (GRE) official results, or equivalent exam For Fall Entry The Artificial Intelligence Master of Engineering and Graduate Certificate programs review applications on a rolling basis. Applications submitted earlier than the deadlines listed below will likely receive an earlier response. ON-CAMPUS Master of Engineering Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by March 15; Decision Notification by April 15; Reply Required by May 1. ONLINE Master of Engineering Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by April 15; Decision Notification by May 15; Reply Required by June 1. ONLINE Certificate Program Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. Application Round 2: Applications received by April 15; Decision Notification by May 15; Reply Required by June 1. to all qualified applicants worldwide. Applications are accepted for the certificate program for the fall semester only, and participants are expected to be working full-time while completing the Certificate program. An application for the AI Foundations for Product Innovation graduate certificate program requires the following: A bachelor’s degree in engineering or science from an accredited institution (transcripts required, including an estimated GPA and a grade scale) Statement of purpose Résumé Two recommendations English Language Testing (TOEFL or IELTS): official results required—international applicants only Video introduction This is a standalone certificate program and does not qualify international students for US visa sponsorship. For students joining the certificate program in the 2023-24 academic year, a limited number of merit-based scholarships are available. All applicants will be automatically considered for the available scholarships based on their application materials. Students who enroll and successfully complete the certificate requirements will have the option to subsequently apply for the online Duke AI for Product Innovation Master of Engineering within four years, and use their certificate courses (12.0 course credits) toward the degree (30.0 course credits), as long as they earn a grade of B or better in each class. Certificate holders who apply to the online Duke AI for Product Innovation Master of Engineering program will additionally be required to provide GRE scores (if required at time of application). GRE scores are optional for the 2023-24 admissions cycle. Applicants who are accepted will then need to complete the remaining requirements for the degree: typically, four electives, Duke AI Master of Engineering (MEng) program is designed to be accessible to participants from a variety of engineering and science backgrounds. Applicants should have: An undergraduate degree in science or engineering (or equivalent technical work experience if your degree is in a non-technical field), and A minimum of one (1) semester of programming (any language), and Sufficient DUOLINGO, IELTS, or TOEFL English Language Testing scores (official result required; international students only), and Two (2) semesters of calculus Prior coursework in probability and statistics and linear algebra is highly encouraged, but not required for admission. Graduate Certificate Program The Duke AI Foundations for Product Innovation Graduate Certificate is designed for working professionals with a technical or scientific background. Participants are expected to be working full-time while completing the Certificate program. Applicants should have: An undergraduate degree in science or engineering (or equivalent technical work experience if your degree is in a non-technical field), and A minimum of one (1) semester of programming (any language), and Sufficient DUOLINGO, IELTS, or TOEFL English Language Testing scores (official result required; international students only), and Two (2) semesters of calculus International applicants - please note that this standalone certificate program does not qualify students for US visa sponsorship. Application Fee US$75 Paid by credit card with your application. Fee waivers » Documentation of your Bachelor's Degree, in engineering or science from an accredited institution: Transcripts (or, for institutions not using a 4.0 system, estimated GPA and grade scale) Other Items: Short Answer Essays Resume for Product Innovation Graduate Certificate A 4-course online program in AI and Machine Learning designed for working professionals Those who have knowledge of AI and machine learning have an advantage. Earn this Duke standalone, credit-bearing, non-degree offering in 15 months by completing: A Python and data science math pre-program boot camp Four (4) technical core courses in AI and machine learning Learn online and pay tuition per course, without the commitment of a traditional degree program. Extensive programming experience is not required, but the following are recommended: An undergraduate degree in science or engineering (or equivalent technical work experience); A minimum of one (1) semester of programming (any language); Two (2) semesters of calculus Students who enroll and complete the certificate will have the option to subsequently apply for the online Duke AI Master of Engineering degree (details below). Join our mailing list to receive updates! COURSES Summer (Pre-Program) - Python & Data Science Math Boot Camp Fall 1 - AIPI 510: Sourcing Data for Analytics Spring 1 - AIPI 520: Modeling Process & Algorithms Summer 1 - AIPI 540: Building Products Using Deep Learning Fall 2 - Departmental Technical Elective Admissions Policies and Practices This certificate program is open to all qualified applicants worldwide. Applications are accepted for the certificate program for the fall semester only, and participants are expected to be working full-time while completing the Certificate program. An application for the AI Foundations for Product Innovation graduate certificate program requires the following: A bachelor’s degree in engineering"
    real_answer_4 = "The deadlines for the online Certificate Program and the on-campus Master of Engineering program at Duke University are similar but not identical. For the online Certificate Program: - Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15. - Application Round 2: Applications received by April 15; Decision Notification by May 15; Reply Required by June 1. For the on-campus Master of Engineering program:- Application Round 1: Applications received by January 15; Decision Notification by March 15; Reply Required by April 15.- Application Round 2: Applications received by March 15; Decision Notification by April 15; Reply Required by May 1.While both programs have a Round 1 deadline in January, the on-campus Master of Engineering program has an additional round with a deadline in March, whereas the online Certificate Program has a second round with a deadline in April. Additionally, the decision notification and reply dates vary slightly between the two programs."

    questions = [q_1, q_2, q_3, q_4]

    
    contexts = [context_1, context_2, context_3, context_4]
    
    actuals = [real_answer_1, real_answer_2, real_answer_3, real_answer_4]

    cosine_similarities = []
    
    answers = []
    
    # Loop through the questions and contexts to get the predicted answers

    for i in range(len(questions)):

        predicted = query(
        {
            "inputs": "",
            "question": f"{questions[i]}",
            "context": f"{contexts[i]}",
            "temp": 0.3,
            "max_tokens": 200
        }
        )
        
        answers.append(predicted)
        
        vectorizer = TfidfVectorizer()
        
        # Calculate the cosine similarity between the predicted and actual answers

        texts = [predicted, actuals[i]]
        
        X = vectorizer.fit_transform(texts)
        
        cosine_sim = cosine_similarity(X)
        
        similarity_score = cosine_sim[0, 1]
        
        cosine_similarities.append(similarity_score)

    max_cosine = max(cosine_similarities)

    min_cosine = min(cosine_similarities)

    avg_cosine = sum(cosine_similarities) / len(cosine_similarities)

    return max_cosine, min_cosine, avg_cosine

def sim_ques_sim():
    """
    Purpose: Function to calculate the cosine similarity for similar questions
    """
    context = "to enroll, and it is ultimately at the instructor’s discretion to determine whether or not AIPI students will be allowed into a particular class. 6. Can I audit classes? Yes, AIPI students may audit courses on a space-available basis with consent of the instructor and the AIPI Director. Audited courses appear on your transcript and you will receive a grade of AD (indicating successful completion) that will not count toward your GPA. Audited courses do not count toward the fulfillment of AIPI degree requirements. 7. How many classes can I audit per semester? AIPI students may audit up to two courses per semester. 8. How many classes should I register for? In order for international students to remain in compliance with their F-1 visas, they must be registered as a full-time student. For visa purposes, full-time enrollment is considered 9.0 credits per semester (equivalent to three 3.0-credit courses). However, a typical full-time load for AIPI Campus students is 12.0 credits per semester (four 3.0-credit courses). Therefore, full-time residential graduate students pay tuition on a flat-rate, per-semester billing basis and are charged the equivalent of 12.0 credits per semester. Most full-time students enroll in at least 12.0 credits per semester. AIPI Campus students take a fixed set of 4 courses in Fall semester (AIPI 510,520,530 and MENG 570). AIPI Online students typically take 2 courses per semester (Fall: AIPI 510,520) and AIPI Certificate students take 1 course/semester (Fall: AIPI 510). Online students pay tuition on a per-credit basis rather than aincoming and continuing students on June 28, 2021, the day that shopping carts open. This list can be accessed using the Simple and Advanced Class Search functions in DukeHub. 3. When can I register for classes? Fall 2021 course registration for all graduate students will begin on July 7, 2021, in DukeHub. Students may continue making changes to their schedule during the Drop/Add period, which will end on September 3. After Drop/Add ends, students may no longer change their Fall 2021 schedules in DukeHub. 4. How do I register for classes? All students register for classes through DukeHub, the self-service application that provides students with an array of information and direct access to academic, financial, and personal data. Access to DukeHub is gained using your NetID and password. For assistance with registration, please see the help guides linked here. 5. What classes outside of the AIPI curriculum can I count toward my technical electives requirement? Approved AIPI electives are listed on the AIPI website. Additionally, students may take technical electives in other programs and departments across campus with approval from both the course instructor and the AIPI Director. Please note that graduate courses outside of AIPI require an instructor’s permission to enroll, and it is ultimately at the instructor’s discretion to determine whether or not AIPI students will be allowed into a particular class. 6. Can I audit classes? Yes, AIPI students may audit courses on a space-available basis with consent of the instructor and the AIPI Director. Audited coursesAcademics 1. What classes are being offered to AIPI students in Fall 2021? In the Fall semester of the AIPI program students take a fixed schedule of courses (electives are taken in the Spring). Students should plan to register for the following courses: - AIPI 503: Bootcamp [0 units] (On-campus, Online MEng, Online Certificate students) - AIPI 510: Sourcing Data for Analytics [3 units] (On-campus, Online MEng, Online Certificate students) - AIPI 520: Modeling Process & Algorithms [3 units] (On-campus & Online MEng students) - AIPI 530: AI in Practice [3 units] (On-campus students) - MENG 570: Business Fundamentals for Engineers [3 units] (On-campus students completing in 12 months) - AIPI 501: Industry Seminar Series [0 units] (On-campus & Online MEng students) The full list of Pratt courses will be made available to students on June 28, 2021, when shopping carts open in DukeHub. 2. When will the list of Fall 2021 courses be available? The list of all Fall 2021 courses offered by the Pratt School of Engineering will be made available to all incoming and continuing students on June 28, 2021, the day that shopping carts open. This list can be accessed using the Simple and Advanced Class Search functions in DukeHub. 3. When can I register for classes? Fall 2021 course registration for all graduate students will begin on July 7, 2021,able to participate in classes and make progress toward their degrees. In the AIPI Program, if you anticipate that you will be able to make it to Durham in time to participate in-person (i.e., by September 22), you should register for the hybrid sections of the courses that you enroll in. You may also enroll in the fully-online section of a course if you know that you will be unable to travel to the United States for the Fall 2021 semester. 34. I hope to arrive in the United States for Fall 2021 classes, but it might be after drop/add ends. What should I do? Students on F-1 visas have until 30 days after the first day of class to arrive in the United States. For Fall 2021, classes begin on August 23, so students must arrive in the United States by September 22. Drop/Add ends on September 3, after which students may not make changes to their Fall 2021 course schedules. Therefore, we recommend that if you anticipate arriving in the United States between September 3 and September 22, you enroll in Fall classes that are being offered according to the Hybrid (i.e., both in-person and online accessible) model. Begin taking these classes online at the beginning of the semester, and if you are able to reach the United States by September 22, you will be able to continue your studies in-person. If you are unable to reach the United States by September 22, you can continue taking your last day that I can register for Fall 2021 classes? Students may make changes to their schedule up to the end of the Drop/Add period, which ends on September 3, 2021. 12. For Campus students, is there a formal process for selecting between the 12-month and 16-month plan? No, there is no specific documentation that must be provided if you choose to extend the AIPI Program to three semesters. However, the earlier you make the decision the better as it will likely affect your selection of courses in Fall and Spring semesters. 13. When do classes start? Fall 2021 classes begin on August 23, 2021. 14. What are the class meeting patterns for AIPI courses? Each course typically meets once per week for 2 hours and 45 minutes. 15. How much time is required outside of class for AIPI courses? AIPI classes are challenging and require significant work outside of class. Each week, you should expect to spend between 8-10 hours per course working outside of class. 16. How will classes be offered in Fall 2021 (e.g., all in-person, online, or a mix of the two)? While Duke will offer classes fully in-person for the Fall 2021 semester, we understand that the COVID-19 pandemic has created travel and visa challenges for many of our international students. AIPI classes will be offered both in-person and online for the Fall 2021 semester. Students who will be physically located in Durham, NC, USA, for the semester are expected to come to class in-person"
    #cosine similarity with one another

    # Questions for evaluation

    q_1 = "How do students from the AIPI register for graduate classes in other departments?"
    q_2 = "What steps must AIPI students take to sign up for graduate-level courses in different disciplines?"
    q_3 = "Can you describe the procedure for AIPI students wishing to enroll in graduate courses not offered by AIPI?"
    q_4 = "What is the enrollment process for AIPI students interested in taking graduate courses from other academic units?"
    q_5 = "How should AIPI students go about enrolling in graduate courses that are part of other departments?"
    
    questions = [q_1, q_2, q_3, q_4, q_5]
    
    answers = []
    
    for q in questions:
        answers.append(query(
        {
            "inputs": "",
            "question": f"{q}",
            "context": f"{context}",
            "temp": 0.3,
            "max_tokens": 200
        }
        ))

    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(answers)
    
    cosine_sim_matrix = cosine_similarity(X)

    averages = []

    for i in range(len(cosine_sim_matrix)):
        row_list = cosine_sim_matrix[i].tolist()
        
        row_list.pop(i)
        
        average = sum(row_list)/len(row_list)

        averages.append(average)
    
    return cosine_sim_matrix, sum(averages)/len(averages)

def lang_structure_sim():
    """
    Purpose: Function to calculate the cosine similarity for questions for 4 structures: 
    1) Indirect Questions
    2) Direct Questions
    3) Compund Questions
    4) Questions with negative connotation
    """
    context = "to enroll, and it is ultimately at the instructor’s discretion to determine whether or not AIPI students will be allowed into a particular class. 6. Can I audit classes? Yes, AIPI students may audit courses on a space-available basis with consent of the instructor and the AIPI Director. Audited courses appear on your transcript and you will receive a grade of AD (indicating successful completion) that will not count toward your GPA. Audited courses do not count toward the fulfillment of AIPI degree requirements. 7. How many classes can I audit per semester? AIPI students may audit up to two courses per semester. 8. How many classes should I register for? In order for international students to remain in compliance with their F-1 visas, they must be registered as a full-time student. For visa purposes, full-time enrollment is considered 9.0 credits per semester (equivalent to three 3.0-credit courses). However, a typical full-time load for AIPI Campus students is 12.0 credits per semester (four 3.0-credit courses). Therefore, full-time residential graduate students pay tuition on a flat-rate, per-semester billing basis and are charged the equivalent of 12.0 credits per semester. Most full-time students enroll in at least 12.0 credits per semester. AIPI Campus students take a fixed set of 4 courses in Fall semester (AIPI 510,520,530 and MENG 570). AIPI Online students typically take 2 courses per semester (Fall: AIPI 510,520) and AIPI Certificate students take 1 course/semester (Fall: AIPI 510). Online students pay tuition on a per-credit basis rather than a Campus students take a fixed set of 4 courses in Fall semester (AIPI 510,520,530 and MENG 570). AIPI Online students typically take 2 courses per semester (Fall: AIPI 510,520) and AIPI Certificate students take 1 course/semester (Fall: AIPI 510). Online students pay tuition on a per-credit basis rather than a flat-rate per-semester basis. 9. What is the limit on credits I can take each semester? AIPI students may take up to 15.0 credits per semester. Full-time residential students on the pay-by-semester basis may take a fifth credit for free each semester (although we generally suggest a maximum of 4 courses as the workload can be intense). Students who attempt to enroll in more than 15.0 credits per semester will not be able to register. 10. What is Drop/Add? What happens during the Drop/Add period? The Drop/Add period occurs after the initial Registration window and continues until the end of the second week of classes. During the Drop/Add period, students may make changes to their schedules through DukeHub. At the end of the Drop/Add period (September 3, 2021), students’ schedules may no longer be changed in DukeHub and can only be changed with permission from their dean. 11. What is the last day that I can register for Fall 2021 classes? Students may make changes to their schedule up to the end of the Drop/Add period, which ends on September 3, 2021. 12. For Campus students, is there a formal process for selecting between the 12-month and 16-month plan? No, there 22. How much does it cost to audit a course? For AIPI students who pay tuition on a pay-by-semester basis (as is the case for all full-time residential AIPI students), there is no charge for auditing a course. For AIPI Online students who pay tuition on a pay-by-credit basis, there is a charge of $535 per audited course. 23. How are tuition and fees assessed for AIPI students? Full-time students in the AIPI Campus Program are automatically set up on a pay-per-semester billing system, meaning they will be charged the equivalent of four separate courses. Students enrolling as AIPI Online and Certificate students will automatically be set up for payments on a per-credit basis. 24. Can I change my tuition billing basis from per-semester to per-credit? Yes, your tuition can be changed from pay-by-semester to pay-by-credit if you are switching to part-time status. (Please note that F-1 visaholders must be enrolled full-time for at least 9.0 credits per semester). If you intend to take less than the typical load (four courses for full-time), please contact Kelsey Liddle (kelsey.liddle@duke.edu), the Pratt Student Records Coordinator, to make this change. The last day for making changes to a student’s billing structure is the last day of Drop/Add in that semester. 25. Is financial aid available to AIPI students? Because the AIPI degree is a professional degree rather than a research degree, most students pay their own tuition costs. For more information on the most common financial aid resources that our students utilize, please track? No, there is not currently a formal process to designate your elective track. We do not require students to rigidly adhere to one elective track. Students may choose electives that fit their professional goals. The elective tracks are meant as guides for students to align and develop skills toward a particular area, and those students who complete a track may list it on their resume. 20. What do I do if I want to change my elective track? If you wish to change your elective track, there is no formal action that you need to take. However, it is a good idea to speak with the program director about your elective course plans, as they can help steer you toward courses that align with your professional aspirations. 21. How can I track my degree progress? Students can track their degree progress using Stellic, a self-service tool that enables students to see which classes they have taken toward their degree and plan for future semesters. Students are strongly encouraged to use Stellic throughout the course of the AIPI Program so that they can stay on track to graduate within the timeframe they choose (two or three semesters). Tuition and Billing 22. How much does it cost to audit a course? For AIPI students who pay tuition on a pay-by-semester basis (as is the case for all full-time residential AIPI students), there is no charge for auditing a course. For AIPI Online students who pay tuition on a pay-by-credit basis, there last day of Drop/Add in that semester. 25. Is financial aid available to AIPI students? Because the AIPI degree is a professional degree rather than a research degree, most students pay their own tuition costs. For more information on the most common financial aid resources that our students utilize, please click the link here. 26. I’m not going to be on campus in the Fall 2021 semester due to COVID-19. Do I still have to pay fees? If you will not be on campus in the Fall 2021 semester due to COVID-19, please contact Kelsey Liddle (kelsey.liddle@duke.edu), the Pratt Student Records Coordinator, regarding student fees. Working While a Student 27. How do I get a Teaching Assistant (TA) position? Teaching assistantships are a common way that AIPI students can work on campus, earn money, and give of their time to the AIPI community. Most often, course instructors approach students who have done well in their course and ask them to TA in a subsequent semester. Other times, students will voice their interest to the instructor to initiate the conversation about a TA position. Toward the beginning of each semester, there are usually a few TA positions to be filled, and an announcement about open positions will be emailed out to students. TA positions are not often available for incoming students, as these positions are typically filled by continuing students who have taken the course before. In order to be on Duke’s payroll, all students must have a Social Security Number"
    q_indirect = " I was wondering if you Could  provide some information on how many credits AIPI students are permitted to enroll in each semester, and what additional advantages are offered to those who are full-time residential students?"
    q_negative = "Isn't there a limit to how few credits AIPI students can take each semester, and if so what is it?, and aren't there special benefits for full-time residential students?"
    q_direct = "How many credits are AIPI students allowed to take each semester, and what additional benefits do full-time residential students receive?"
    q_compound = "How many credits can AIPI students take each semester? Is there a hard limit? Additionally, what special benefits are provided to students who choose to live on campus full-time?"

    questions = [q_indirect, q_negative, q_direct, q_compound]
    answers = []
    for q in questions:
        answers.append(query(
        {
            "inputs": "",
            "question": f"{q}",
            "context": f"{context}",
            "temp": 0.3,
            "max_tokens": 200
        }
        ).lower())

    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(answers)
    
    cosine_sim_matrix = cosine_similarity(X)

    averages = []
    
    for i in range(len(cosine_sim_matrix)):
        row_list = cosine_sim_matrix[i].tolist()

        row_list.pop(i)

        average = sum(row_list)/len(row_list)

        averages.append(average)
    
    return sum(averages)/len(averages)


def tokenize(response):
    """
    Purpose: Function to tokenize the response
    Input: response - the response to be tokenized
    """
    return set(word_tokenize(response.lower()))  # Using set to remove duplicates and ignore order


def calculate_recall_and_precision(generated, actual):
    """
    Purpose: Function to calculate the recall, precision and F1 score
    Input: generated - the generated response
    Input: actual - the actual response
    """
    generated_tokens = tokenize(generated)
    actual_tokens = tokenize(actual)

    correct_tokens = generated_tokens.intersection(actual_tokens)
    recall = len(correct_tokens) / len(actual_tokens) if actual_tokens else 0
    precision = len(correct_tokens) / len(generated_tokens) if generated_tokens else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return recall, precision, f1_score

def recall_precision_f1():
    """
    Purpose: Function to calculate the recall, precision and F1 score for the validation dataset
    """

    dataset = load_dataset("suneeln-duke/duke_qac_v3")

    val_df = pd.DataFrame(dataset['val'])

    val_questions = val_df['Question'].tolist()
    
    val_contexts = val_df['Context'].tolist()
    
    val_answers = val_df['Answer'].tolist()
    
    generated_val_answers = []

    for i in range(len(val_questions)):
        generated_val_answers.append(query(
        {
            "inputs": "",
            "question": f"{val_questions[i]}",
            "context": f"{val_contexts[i]}",
            "temp": 0.3,
            "max_tokens": 200
        }
        ).lower())
    
    recalls = []
    
    precisions = []
    
    f1s = []
    
    for i in range(len(generated_val_answers)):
        recall, precision, f1 = calculate_recall_and_precision(generated_val_answers[i], val_answers[i])
        
        recalls.append(recall)
        
        precisions.append(precision)
        
        f1s.append(f1)
    
    return sum(recalls)/len(recalls), sum(precisions)/len(precisions), sum(f1s)/len(f1s), generated_val_answers, val_df

def eval():
    """
    Purpose: Function to evaluate the model based on the metrics defined
    """

    max_stringent, min_stringent, avg_stringent = stringent_acc()

    cosine_sim_matrix, similar_question_consistency = sim_ques_sim()

    lang_structure_testing = lang_structure_sim()

    recall, precision, f1, generated_val_answers, val_df = recall_precision_f1()

    text = f"""
    Stringent Factual Accuracy:
    Average Cosine Similarity: {avg_stringent}
    Maximum Cosine Similarity: {max_stringent}
    Minimum Cosine Similarity: {min_stringent}

    Similar Question Consistency: {similar_question_consistency}

    Language Structure Testing: {lang_structure_testing}

    Recall: {recall}

    Precision: {precision}

    F1 Score: {f1}
    """
    val_df['Predicted Answers'] = generated_val_answers

    # write the metrics to a file

    f = open("./data/output/metrics.txt","w+")

    f.write(text)

    val_df.to_csv("./data/output/llm_responses.csv")
