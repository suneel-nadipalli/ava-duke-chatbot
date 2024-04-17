
import csv
import openai

# Define the OpenAI API key here
openai.api_key = ''

def read_questions_from_csv(file_path):
    """
    Purpose: Read questions from a CSV file.
    Input: file_path - the path to the CSV file containing the questions.
    """
    questions = []
    # Here we loop through the text file and read the questions
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # For each row within the question, we append the question to the questions list
        for row in reader:
            # Append the question to the questions list
            questions.append(row[0])
    return questions

def rephrase_questions(questions, target_total, variations_per_question=2): 
    """
    Purpose: Rephrase the questions using the GPT-3.5 model.
    Input: questions - the list of questions to rephrase
    Input: target_total - the target total number of questions to generate
    Input: variations_per_question - the number of variations to generate per question
    """ 
    rephrased_questions = []
    # Here we calculate the number of questions needed to reach the target total
    # There are roughly 328 questions in the original questions list
    needed = max(target_total - len(questions), 0)

    for question in questions:
        # Default count variable to keep track of the number of rephrased questions
        count = 0
        # While the count is less than the variations per question and the length of the rephrased questions is less than the needed questions
        # we generate different ways to ask the question
        while count < variations_per_question and len(rephrased_questions) < needed:
            # This prompt is used to generate different ways to ask the question
            prompt_text = f"Generate {variations_per_question} different ways to ask the following question: {question}"
            try:
                response = openai.Completion.create(
                    # We use the GPT-3.5-turbo-instruct model to generate the rephrased questions
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt_text,
                    max_tokens=100,
                    n=variations_per_question,
                    stop=None,
                    temperature=0.8
                )
                # For each rephrased question in the response, we append it to the rephrased questions list
                for rephrased in response.choices[0].text.strip().split('\n'):
                    # If the rephrased question is not empty and is not the same as the original question, we append it to the rephrased questions list
                    if rephrased.strip() and rephrased.strip() != question:
                        rephrased_questions.append(rephrased.strip())
                        # We increase the count variable tracker by 1 each time we add a rephrased question
                        count += 1
                        if len(rephrased_questions) >= needed:
                            break
            except Exception as e:
                print(f"Error processing question: {str(e)}")

    return questions + rephrased_questions 

def save_questions_to_csv(questions, output_path):
    """
    Purpose: Save the questions to a CSV file.
    Input: questions - the list of questions to save
    Input: output_path - the path to the CSV file to save the questions
    """
    # We open a new CSV file and write the questions to the file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # We use the csv writer to write the questions to the file
        writer = csv.writer(csvfile)
        # For all questions in the questions list, we write the question to the file
        for question in questions:
            writer.writerow([question])

if __name__ == "__main__":
    # Here we define the input and output file paths
    input_file_path = '/content/questions.csv'  
    output_file_path = '/content/rephrased_questions.csv'  

    # Here we use the functions defined above to rephrase the questions
    original_questions = read_questions_from_csv(input_file_path)
    # We generate 700 rephrased questions
    rephrased_questions = rephrase_questions(original_questions, 700) 
    # We save the rephrased questions to a CSV file 
    save_questions_to_csv(rephrased_questions, output_file_path)
    # We print the total number of questions generated
    print(f"Total questions generated: {len(rephrased_questions)}")
