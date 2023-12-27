from langchain.prompts import PromptTemplate
from typing import Tuple
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import PersonIntel, person_intel_parser
from third_party.linkedin import scrape_linkedin_profile

name="Vedanth Vasishth"

def ice_break(name:str)-> Tuple[PersonIntel,str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    summary_template = """
       given the Linkedin information {information} about a person I want you to create:
       1. a short summary
       2. two interesting facts about them
       3. A topic that may interest them
       4. 2 creative Ice breakers to open a conversation with them
      \n{format_instructions}
       """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template, partial_variables={"format_instructions":person_intel_parser.get_format_instructions()})

    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo") # temperature=1

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)


   # result= chain.run(information=linkedin_data)

   # return person_intel_parser.parse(result),linkedin_data.get("profile_pic_url")
    result= chain.run(information=linkedin_data)
    print(result)
    return result

if __name__== '__main__':
    print("Hello Langchain")
    result=ice_break(name="Vedanth Vasishth")

# def ice_break(name: str) -> Tuple[PersonIntel, str]:
#     linkedin_profile_url = linkedin_lookup_agent(name=name)
#     linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
#
#     summary_template = """
#        given the Linkedin information {information} about a person I want you to create:
#        1. a short summary
#        2. two interesting facts about them
#        3. A topic that may interest them
#        4. 2 creative Ice breakers to open a conversation with them
#       \n{format_instructions}
#        """
#
#     summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template,
#                                              partial_variables={"format_instructions": person_intel_parser.get_format_instructions()})
#
#     llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")
#     chain = LLMChain(llm=llm, prompt=summary_prompt_template)
#
#     # Run the chain
#     result = chain.run(information=linkedin_data)
#
#     # Ensure that result has the expected structure
#     if len(result) >= 2:
#         person_intel, _ = result[:2]  # Extract the first two values
#     else:
#         # Handle the case where there are not enough values in the result
#         person_intel = result[0] if result else None
#
#     # Extract the profile_pic_url from linkedin_data
#     profile_pic_url = linkedin_data.get("profile_pic_url", "")
#
#     return person_intel, profile_pic_url


