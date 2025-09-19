# app.py
import squarify
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc

# --- Set up the Streamlit page ---
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Stack Overflow Survey Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions from Your Notebook ---
# These functions should be placed at the top of your script.
@st.cache_data
def load_data():
    # Use a spinner to show progress while loading and cleaning data
    with st.spinner("Loading and processing data..."):
        # Make sure 'survey_results_public.csv' is in the same folder as this script.
        df = pd.read_csv('survey_results_public.csv')
        # --- Data Cleaning ---
        # 1. Drop rows with outliers in 'Age'
        df.drop(df[df.Age < 10].index, inplace=True)
        df.drop(df[df.Age > 100].index, inplace=True)

        # 2. Clean 'Gender' column
        df.where(~(df.Gender.str.contains(';', na=False)), np.nan, inplace=True)

        # 3. Convert required columns to numeric
        df['Age1stCode'] = pd.to_numeric(df.Age1stCode, errors='coerce')
        df['YearsCode'] = pd.to_numeric(df.YearsCode, errors='coerce')
        df['YearsCodePro'] = pd.to_numeric(df.YearsCodePro, errors='coerce')

        # 4. Drop rows with impossible 'WorkWeekHrs'
        df.drop(df[df.WorkWeekHrs > 140].index, inplace=True)

        # 5. Create 'is_English_speaking' column
        english_speaking_countries = ['United States', 'United Kingdom', 'Canada', 'Australia', 'New Zealand', 'Ireland', 'South Africa']
        df['is_English_speaking'] = df['Country'].isin(english_speaking_countries)

        # 6. Create 'AgeGroup' column
        bins = [0, 10, 18, 30, 45, 60, np.inf]
        labels = ['Less than 10 Years', '10-18 Years', '18-30 Years', '30-45 Years', '45-60 Years', 'Older than 60 Years']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

        # 7. Create 'HasCollegeDegree' column
        college_degrees = [
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
        'Professional degree (JD, MD, etc.)',
        'Associate degree (A.A., A.S., etc.)',
        'Other doctoral degree (Ph.D., Ed.D., etc.)',
        ]
        df['HasCollegeDegree'] = np.where(df['EdLevel'].isin(college_degrees), 'Some College Degree', 'No College Degree')
        
        # 8. Create 'EmploymentType' column
        employment_type_map = {
            'Student': 'Enthusiast',
            'Not employed, but looking for work': 'Enthusiast',
            'Employed full-time': 'Professional',
            'Independent contractor, freelancer, or self-employed': 'Professional',
            'Employed part-time': 'Professional',
            'Not employed, and not looking for work': 'Other',
            'Retired': 'Other'
        }
        df['EmploymentType'] = df.Employment.map(employment_type_map)

        # 9. Create 'Continent' column
        df['Continent'] = df['Country'].apply(get_continent)

        # 10. Create 'ExperienceGroup' column
        df['ExperienceGroup'] = np.where(
            df.YearsCodePro >= 10,
            '10+ Years',
            'Less Than 10 Years'
        )
    return df

@st.cache_data
def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    for idex, value in col_series[col_series.notnull()].items():
        for option in value.split(';'):
            if option not in result_df.columns:
                options.append(option)
                result_df[option] = False
            result_df.at[idex, option] = True
    return result_df[options]

@st.cache_data
def get_continent(country_name):
    # This function safely maps country names to continents.
    try:
        if not isinstance(country_name, str):
            return 'Not Found'
        country_alpha2 = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except KeyError:
        return 'Not Found'
    except Exception:
        return 'Not Found'

# --- Load the data ---
survey_df = load_data()
dev_type_df = split_multicolumn(survey_df.DevType)
languages_worked_df = split_multicolumn(survey_df.LanguageWorkedWith)
languages_interested_df = split_multicolumn(survey_df.LanguageDesireNextYear)

# --- Sidebar for navigation ---
st.sidebar.title('Explore by Topic ->')
topic = st.sidebar.radio(
    'Select a topic:',
    [
        'Overview',
        'Survey Demographics',
        'Coding Experience',
        'Languages & Tools',
        'Work & Employment',
        'Compensation'
    ]
)
# --- Main content based on page selection ---
st.title("Stack Overflow Developer Survey 2019 Analysis")
st.markdown("A deep dive into the 2019 developer survey data, exploring key trends, demographics, and compensation insights.")
st.markdown("---")

if topic == 'Overview':
    st.header('Survey Overview')
    st.subheader('The State of the Modern Developer: A Data-Driven Analysis of Skills, Salaries, Compensation and Satisfaction')
    st.write('This dashboard explores key findings from the Stack Overflow Developer Survey 2019.')
    st.info(f'Total respondents in the survey: **{len(survey_df)}**')
    st.dataframe(survey_df.head(), use_container_width=True)

elif topic == 'Survey Demographics':
    st.header('Survey Demographics')
    st.markdown('Get an overview of who participated in the survey — countries, ages, education, and more.')

    # 1. Number of respondents by country
    st.info('See which countries contributed the most voices to the survey.')
    top_countries = survey_df.Country.value_counts().head(15)
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=top_countries.index, y=top_countries, hue=top_countries.index, palette='viridis', ax=ax )

    ax.set_title('Where do you live?')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Respondents')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)    

    #2. Percentage of Responses by English-speaking vs Non-English-speaking Countries
    st.info('Compare how responses differ between English-speaking and non-English-speaking countries.')
    languages_df = pd.read_csv('https://raw.githubusercontent.com/jvcasillas/worldlanguages/refs/heads/main/data-raw/languages_by_country_2022-12-13.csv')
    english_speaking_countries = languages_df[languages_df['official_language'].str.contains('English')]['country_region'].unique()
    survey_df['is_English_speaking'] = survey_df['Country'].apply(lambda x : 'English-Speaking' if x in english_speaking_countries else 'Non-English-Speaking')
    percentages_responses = survey_df.is_English_speaking.value_counts(normalize= True) * 100    
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=percentages_responses.index, y=percentages_responses, hue=percentages_responses.index, palette='viridis', ax=ax)

    ax.set_title('Percentage of Responses by Country Type')
    ax.set_xlabel('Country Type')
    ax.set_ylabel('Percentage')
    for index, value in enumerate(percentages_responses):
      ax.text(index, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')
    ax.set_ylim(0, 100)
    st.pyplot(fig)    

    # 3. Age Distribution of Respondents
    st.subheader('Explore the range of ages among developers worldwide.')
    sns.set_style('darkgrid')

    fig, ax = plt.subplots(figsize=(12,6))
    ax.hist(survey_df.Age, bins=np.arange(10, 100, 5), color='purple')

    ax.set_title('Age Distribution of Respondents')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Respondents')
    st.pyplot(fig)   

    # 4. Age Group Distribution
    st.info('Break down respondents into age groups to spot trends by generation.')
    
    bins = [0, 10, 18, 30, 45, 60, np.inf]
    labels = ['Less than 10 Years', '10-18 Years', '18-30 Years', '30-45 Years', '45-60 Years', 'Older than 60 Years']

    survey_df['AgeGroup'] = pd.cut(
        survey_df['Age'],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    agegroup_counts = survey_df.AgeGroup.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(
        x=agegroup_counts.index, 
        y=agegroup_counts.values, 
        hue=agegroup_counts.index, 
        palette='turbo',
        ax=ax
    )
    ax.set_title('The results of the survey for different age groups')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Number of Respondents')
    plt.xticks(rotation=45, ha='right')
    
    for index, values in enumerate(agegroup_counts.values):
      ax.text(index, values + 50, str(values), ha='center', va='bottom')
      
    plt.tight_layout()
    st.pyplot(fig)

    # 5. Gender Distribution
    st.info('Visualize the gender diversity represented in the developer community')
    gender_counts = survey_df.Gender.value_counts(dropna=False)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.pie(
        gender_counts, 
        labels=gender_counts.index, 
        autopct='%1.1f%%', 
        startangle=180, 
        colors=sns.color_palette('dark')
    )
    ax.set_title('Which of the following describe you, if any? Please check all that apply. \n If you prefer not to answer, you may leave this question blank.')
    st.pyplot(fig)

    # 6. Education Levels by Gender
    st.info('Examine how education levels vary across different genders.')
    filtered_gender_df = survey_df.dropna(subset=['Gender'])
    filtered_gender_df = filtered_gender_df[~filtered_gender_df.Gender.str.contains(';', na=False)]
    
    fig, ax = plt.subplots(figsize=(11,6))
    sns.countplot(
        x='EdLevel',
        hue='Gender',
        data=filtered_gender_df,
        palette='Set2',
        ax=ax
    )
    ax.set_title('Education Levels by Gender')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Number of Respondents')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # 7. Salary Analysis by Gender
    st.info('Compare median salaries by gender to uncover patterns in compensation.')
    salary_df = survey_df.dropna(subset=['ConvertedComp', 'Gender']).copy()
    salary_df = salary_df[salary_df['ConvertedComp'] > 1000]
    salary_df['Gender'] = salary_df['Gender'].apply(
        lambda x: 'Man' if x=='Man' else ('Woman' if x == 'Woman' else 'Other')
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(
        x='Gender',
        y='ConvertedComp',
        data=salary_df,
        palette='magma',
        ax=ax
    )
    ax.set_title('Median Salary Comparison by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Annual Compensation (USD)')
    ax.set_yscale('log')
    st.pyplot(fig)

elif topic == 'Coding Experience':
    st.header('Coding Experience')
    st.write('This section explores how and when people started their coding journey and their formal education.')
    with st.expander("Explore when developers started coding and how their experience varies.", expanded=False): # Here is the st.expander call
        
        # NEW GRAPH 2: Age at First Code
        st.info('Discover the age when developers first dipped their toes into programming.')
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(data=survey_df, x='Age1stCode', kde=True, palette='turbo', ax=ax);
        ax.set_title('At what age did you write your first line of code or program? (e.g., webpage, Hello World, Scratch project)');
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            # NEW GRAPH 1: Age vs. Professional Experience
            st.container(border=False)
            st.info('Compare starting age with years of professional coding experience')
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(x='Age', y='YearsCodePro', hue='Hobbyist', data=survey_df, ax=ax);
            ax.set_xlabel('Age in years');
            ax.set_ylabel("Years of professional coding experience");
            ax.set_title('Age vs. Years of Professional Coding Experience');
            st.pyplot(fig)

        with col2:
            st.container(border=False)    
            # NEW GRAPH 3: Professional Experience by Gender
            st.info('Explore how coding experience differs across genders.')
            proExp_df = survey_df.dropna(subset=['YearsCodePro', 'Gender']).copy()
            proExp_df['Gender_Cleaned'] = proExp_df['Gender'].apply(
                lambda x: 'Man' if x == 'Man' else ('Woman' if x=='Woman' else 'Other')
            )
            fig, ax = plt.subplots(figsize=(10,6));
            sns.boxplot(
                hue='Gender_Cleaned',
                y='YearsCodePro',
                data=proExp_df,
                palette='plasma',
                ax=ax
            );
            ax.set_title('Professional Experience by Gender')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Years of Professional Coding Experience')
            st.pyplot(fig)

    with st.expander("See how education shapes developers’ careers and skills.", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.container(border=False)
            # 1. Highest Formal Education Countplot
            st.info('See the most common levels of formal education among developers.')
            fig, ax = plt.subplots(figsize=(12,8))
            sns.countplot(y=survey_df.EdLevel, ax=ax, palette='Set3');
            ax.set_title('Highest Formal Education Level Completed')
            ax.set_xlabel('Number of Respondents')
            ax.set_ylabel(None)
            plt.xticks(rotation=75, ha='right')
            st.pyplot(fig)
        
        with col2:
            st.container(border=False)
            # 2. Highest Formal Education (Percentages)
            st.info('View education levels as a percentage breakdown for clearer comparisons')
            edLevel_percentages = survey_df.EdLevel.value_counts(normalize=True).sort_values(ascending=False) * 100
            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(x=edLevel_percentages.values, y=edLevel_percentages.index, palette='viridis', ax=ax)
            ax.set_title('Which of the following best describes the highest level \n of formal education that you’ve completed? (In percentages)')
            ax.set_xlabel('Percentage %')
            ax.set_ylabel('Education Level')
            for index, value in enumerate(edLevel_percentages.values):
                ax.text(value, index, f'{value:.1f}%', va='center')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.container(border=False)
        # 3. Education Level Percentages by Gender
        st.info('Compare education attainment across different genders.')
        
        gender_map = {
            'Man': 'Man',
            'Woman': 'Woman',
            'Non-binary, genderqueer, or gender non-conforming': 'Other'
        }

        filtered_df = survey_df.dropna(subset=['EdLevel','Gender']).copy()
        filtered_df['GenderCleaned'] = filtered_df['Gender'].map(gender_map).fillna('Other')

        edlevel_by_gender = pd.crosstab(
            index=filtered_df['EdLevel'],
            columns=filtered_df['GenderCleaned'],
            normalize='index'
        )*100
        
        fig, ax = plt.subplots(figsize=(15,8))
        edlevel_by_gender.plot(kind='bar', stacked=False, ax=ax)
        ax.set_title('Education Level Percentages by Gender')
        ax.set_xlabel('Education Level')
        ax.set_ylabel('Percentage %')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.tight_layout()
        st.pyplot(fig)

        # 4. Undergraduate Major Percentages
        st.container(border=False)
        st.info('Check which undergraduate majors are most common among developers.')
        undergrad_pct = survey_df.UndergradMajor.value_counts()*100 / survey_df.UndergradMajor.count()
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=undergrad_pct, y=undergrad_pct.index, hue=undergrad_pct.index, palette='turbo', ax=ax)
        ax.set_title('What was your primary field of study?')
        ax.set_xlabel('Percentage')
        ax.set_ylabel('Undergraduate Major')
        st.pyplot(fig)

        # 5. Importance of Formal Education by Degree Status
        st.container(border=False)
        st.info('Explore how developers perceive the importance of formal education.')
        
        college_degrees = [
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
        'Professional degree (JD, MD, etc.)',
        'Associate degree (A.A., A.S., etc.)',
        'Other doctoral degree (Ph.D., Ed.D., etc.)',
        ]
        survey_df['HasCollegeDegree'] = np.where(survey_df['EdLevel'].isin(college_degrees), 'Some College Degree', 'No College Degree')
        plot_df = survey_df.dropna(subset=['NEWEdImpt', 'HasCollegeDegree'])

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.countplot(
            hue='HasCollegeDegree',
            y='NEWEdImpt',
            data=plot_df,
            order=plot_df['NEWEdImpt'].value_counts().index,
            palette='magma',
            ax=ax
        )
        ax.set_title('Importance of formal educaton by college degree status')
        ax.set_xlabel('Number of respondents')
        ax.set_ylabel('Importance of formal education')
        ax.legend(title='College Degree Status')
        plt.tight_layout()
        st.pyplot(fig)

        #7. Opinions
        st.info('Visualize developers’ opinions on education in percentage form')
        ed_impt_by_degree = pd.crosstab(
        index=plot_df['NEWEdImpt'],
        columns=plot_df["HasCollegeDegree"],
        normalize='index'
        ) * 100
        st.dataframe(ed_impt_by_degree.style.format('{:.2f}%'), use_container_width=True)


        # 6. Hobbyist Analysis (New)
        st.container(border=False)
        st.info('Find out how many developers code just for fun outside of work.')
        hobbyist_counts = survey_df.Hobbyist.value_counts(normalize=True)*100
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(
            hobbyist_counts,
            labels=hobbyist_counts.index,
            autopct='%1.1f%%',
            startangle=100,
            colors=sns.color_palette('turbo')
        )
        ax.set_title('Are you a Hobbyist?')
        st.pyplot(fig)



elif topic == 'Languages & Tools':
    st.container(border=False)
    st.header('Languages & Tools')
    st.markdown('Dive into the world of code: explore the languages and technologies shaping developers’ day-to-day work.')

    # 1. Languages Used in the Past Year
    st.container(border=False)
    st.info('See which programming languages dominated developers’ toolkits over the last year.')
    
    languages_worked_df = split_multicolumn(survey_df.LanguageWorkedWith)
    languages_worked_percentages = languages_worked_df.mean().sort_values(ascending=False)*100
    
    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(
        x=languages_worked_percentages.values,
        y=languages_worked_percentages.index,
        hue=languages_worked_percentages.index,
        palette='hls',
        ax=ax,
        legend=False
    )
    ax.set_title('Languages Used in the Past Year')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Language')
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        # 15. Most Loved Languages
        st.container(border=False)
        st.info('Discover the languages developers are most passionate about — the ones they truly enjoy working with.')
        languages_loved_df = languages_worked_df & languages_interested_df
        languages_loved_percentages = (languages_loved_df.sum() *100 / languages_worked_df.sum()).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10,6));
        sns.barplot(
            x=languages_loved_percentages.values,
            y=languages_loved_percentages.index,
            hue=languages_loved_percentages.index,
            palette='mako',
            ax=ax,
            legend=False
        );
        ax.set_title('Languages Loved');
        ax.set_xlabel('Percentage of Users Who Worked With and Desired It');
        ax.set_ylabel('Language');
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
    # 16. Most Dreaded Languages
        st.container(border=False)
        st.info('Uncover the languages developers would rather avoid — \ntools that spark more frustration than joy')
        languages_dreaded_df = languages_worked_df & ~languages_interested_df
        languages_dreaded_percentages = (languages_dreaded_df.sum() * 100 / languages_worked_df.sum()).sort_values(ascending=False)
    
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(
            x=languages_dreaded_percentages.values,
            y=languages_dreaded_percentages.index,
            hue=languages_dreaded_percentages.index,
            palette='turbo',
            ax=ax,
            legend=False
        )
        ax.set_title('Languages Dreaded')
        ax.set_xlabel('Percentage of Users Who Worked With It But Did Not Desire It');
        ax.set_ylabel('Language')
        plt.tight_layout()
        st.pyplot(fig)

    # 8. Languages Interested in Learning Next Year (New)
    st.container(border=False)
    st.info('Peek into the future: which languages are developers most excited to learn and add to their arsenal next year?')
    languages_interested_df = split_multicolumn(survey_df.LanguageDesireNextYear)
    languages_interested_percentages = languages_interested_df.mean().sort_values(ascending=False)*100

    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(
        x=languages_interested_percentages.values,
        y=languages_interested_percentages.index,
        hue=languages_interested_percentages.index,
        palette='magma',
        ax=ax,
        legend=False
    )
    ax.set_title('Languages Interested in Learning Next Year')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Language')
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Compare the most common programming languages across different roles, industries, and developer communities.", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            # 2. Most Common Languages Among Enthusiasts (Students)
            st.container(border=False)
            st.info('Most Common Languages Among Enthusiasts')
            student_mask = survey_df.EmploymentType == 'Enthusiast'
            student_languages_df = languages_worked_df[student_mask]
            student_languages_percentages = student_languages_df.mean().sort_values(ascending=False)*100
            
            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=student_languages_percentages.values,
                y=student_languages_percentages.index,
                hue=student_languages_percentages.index,
                palette='cividis',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages Among  [Enthusiasts]')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # 3. Languages Used by Professionals (New)
            st.container(border=False)
            st.info('Most Common Languages Among Professionals')
            professionals_mask = survey_df.EmploymentType == 'Professional'
            professional_languages_df = languages_worked_df[professionals_mask]
            professional_languages_percentages = professional_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=professional_languages_percentages.values,
                y=professional_languages_percentages.index,
                hue=professional_languages_percentages.index,
                palette='mako',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages Among Professionals')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            # 4. Languages Among Non-Front-end Developers (New)
            st.container(border=False)
            st.info('Most Common Languages Among Non-Front-end Developers')
            non_frontend_mask = ~dev_type_df['Developer, front-end']
            non_frontend_languages_df = languages_worked_df[non_frontend_mask]
            non_frontend_languages_percentages = non_frontend_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=non_frontend_languages_percentages.values,
                y=non_frontend_languages_percentages.index,
                hue=non_frontend_languages_percentages.index,
                palette='Set1',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages Among Non-Front-end Developers')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # 5. Languages in Data Science Fields (New)
            st.container(border=False)
            st.info('Most Common Languages in Data Science Fields')
            ds_roles = [
                'Data scientist or machine learning specialist',
                'Engineer, data',
                'Data or business analyst'
            ]
            ds_mask = dev_type_df[ds_roles].any(axis=1)
            ds_languages_df = languages_worked_df[ds_mask]
            ds_languages_percentages = ds_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=ds_languages_percentages.values,
                y=ds_languages_percentages.index,
                hue=ds_languages_percentages.index,
                palette='Set2',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages Among Data Science Professionals')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            # 6. Languages Among Developers Older Than 35 (New)
            st.container(border=False)
            st.info('Most Common Languages Among Developers Older Than 35')
            older_devs_mask = survey_df.Age >= 35
            older_devs_languages_df = languages_worked_df[older_devs_mask]
            older_devs_percentages = older_devs_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=older_devs_percentages.values,
                y=older_devs_percentages.index,
                hue=older_devs_percentages.index,
                palette='pastel',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages Among Developers Older Than 35')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # 7. Languages in Your Home Country (India) (New)
            st.container(border=False)
            st.info('Most Common Languages in Your Home Country (India)')
            india_mask = survey_df.Country == 'India'
            india_languages_df = languages_worked_df[india_mask]
            india_languages_percentages = india_languages_df.mean().sort_values(ascending=False)*100
            
            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(
                x=india_languages_percentages.values,
                y=india_languages_percentages.index,
                hue=india_languages_percentages.index,
                palette='dark',
                ax=ax,
                legend=False
            )
            ax.set_title('Most Common Languages in Your Home Country (India)')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)
    
    with st.expander("Find out which languages professionals dream of mastering, broken down by job roles and specialties.", expanded=False):

        col1, col2 = st.columns(2)
        with col1:
            # 9. Languages of Interest for Students (New)
            st.container(border=False)
            st.info('Most Desired Languages Among Enthusiasts')
            student_mask = survey_df.EmploymentType == 'Enthusiast'
            student_languages_df = languages_interested_df[student_mask]
            student_languages_percentages = student_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            squarify.plot(
                sizes=student_languages_percentages.values,
                label=student_languages_percentages.index,
                color=sns.color_palette('turbo', len(student_languages_percentages)),
                alpha=0.8,
                ax=ax
            )
            ax.set_title('Most Desired Languages Among Enthusiasts')
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            # 10. Languages of Interest for Professionals (New)
            st.container(border=False)
            st.info('Most Desired Languages Among Professionals')
            professionals_mask = survey_df.EmploymentType == 'Professional'
            professional_languages_df = languages_interested_df[professionals_mask]
            professionals_languages_percentages = professional_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(12,8))
            ax.hlines(
                y=professionals_languages_percentages.index,
                xmin=0,
                xmax=professionals_languages_percentages.values,
                color='gray'
            )
            ax.plot(
                professionals_languages_percentages.values,
                professionals_languages_percentages.index,
                'o', color='crimson'
            )
            ax.set_title('Most Desired Languages Among Professionals')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            # 11. Languages of Interest Among Non-Front-end Developers (New)
            st.container(border=False)
            st.info('Most Desired Languages Among Non-Front-end Developers')
            non_front_end_mask = ~dev_type_df['Developer, front-end']
            non_front_end_languages_df = languages_interested_df[non_front_end_mask]
            non_front_end_languages_percentages = non_front_end_languages_df.mean().sort_values(ascending=False)*100

            fig, ax = plt.subplots(figsize=(10,8))
            sns.stripplot(
                x=non_front_end_languages_percentages.values,
                y=non_front_end_languages_percentages.index,
                hue=non_front_end_languages_percentages.index,
                palette='dark:dodgerblue',
                size=10,
                ax=ax,
                legend=False
            )
            ax.set_title('Most Desired Languages Among Non-Front-end Developers')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # 12. Languages of Interest in Data Science Fields (New)
            st.container(border=False)
            st.info('Most Desired Languages in Data Science Fields')
            ds_roles = [
                'Data scientist or machine learning specialist',
                'Engineer, data',
                'Data or business analyst'
            ]
            ds_mask = dev_type_df[ds_roles].any(axis=1)
            ds_languages_df = languages_interested_df[ds_mask]
            ds_languages_percentages = ds_languages_df.mean().sort_values(ascending=False)*100
            top_10_languages = ds_languages_percentages.head(10)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.pie(
                top_10_languages.values,
                labels=top_10_languages.index,
                autopct='%1.1f%%',
                startangle=180,
                colors = sns.color_palette('plasma', len(top_10_languages))
            )
            ax.set_title('Most Desired Languages Among Data Science Professionals')
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            # 13. Languages of Interest Among Developers Older Than 35 (New)
            st.container(border=False)
            st.info('Most Desired Languages Among Developers Older Than 35')
            older_devs_mask = survey_df.Age >=35
            older_devs_languages_df = languages_interested_df[older_devs_mask]
            older_devs_percentages = older_devs_languages_df.mean().sort_values(ascending=False)*100
            top_15_interested_languages = older_devs_percentages.head(15)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.hlines(
                y=top_15_interested_languages.index,
                xmin=0,
                xmax=top_15_interested_languages.values,
                color='gray'
            )
            sns.scatterplot(
                y=top_15_interested_languages.index,
                x=top_15_interested_languages.values,
                s=150,
                color='steelblue',
                alpha=0.8,
                ax=ax
            )
            ax.set_title('Most Desired Languages Among Developers Older Than 35')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:    
            # 14. Languages of Interest in Your Home Country (India) (New)
            st.container(border=False)
            st.info('Most Desired Languages in Your Home Country (India)')
            india_mask = survey_df.Country == 'India'
            india_languages_df = languages_interested_df[india_mask]
            india_languages_percentages = india_languages_df.mean().sort_values(ascending=False)*100
            top15_indian_languages = india_languages_percentages.head(15)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.hlines(
                y=top15_indian_languages.index,
                xmin=0,
                xmax=top15_indian_languages.values,
                color='gray'
            )
            sns.scatterplot(
                y=top15_indian_languages.index,
                x=top15_indian_languages.values,
                s=150,
                color='red',
                alpha=0.8,
                ax=ax
            )
            ax.set_title('Most Desired Languages in Your Home Country (India)')
            ax.set_xlabel('Percentage')
            ax.set_ylabel('Language')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)


elif topic == 'Work & Employment':
    st.container(border=False)
    st.header('Work & Employment')
    st.markdown('Explore jobs, work hours, and satisfaction among developers.')

    # 1. Employment Type Distribution
    st.container(border=False)
    st.info('See how developers are employed worldwide.')
    
    employment_percentages = (survey_df.Employment.value_counts(normalize=True, ascending=True) * 100)
    
    fig, ax = plt.subplots(figsize=(10,6))
    employment_percentages.plot(kind='barh', color='g', ax=ax)
    
    ax.set_title('Which of the following best describes your current employment status?')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Employment Type')
    
    st.pyplot(fig)

    # 2. Categorical Employment Type Distribution
    st.container(border=False)
    st.info('Compare employment types across categories.')
    
    employment_type_map = {
        'Student': 'Enthusiast',
        'Not employed, but looking for work': 'Enthusiast',
        'Employed full-time': 'Professional',
        'Independent contractor, freelancer, or self-employed': 'Professional',
        'Employed part-time': 'Professional',
        'Not employed, and not looking for work': 'Other',
        'Retired': 'Other'
    }

    survey_df['EmploymentType'] = survey_df.Employment.map(employment_type_map)
    employment_type_counts = survey_df['EmploymentType'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=employment_type_counts.index,
        y=employment_type_counts.values,
        palette='magma',
        ax=ax
    )
    ax.set_title('Distribution of Employment Types into Categories')
    ax.set_xlabel('Employment Type')
    ax.set_ylabel('Number of respondents')

    for index, values in enumerate(employment_type_counts):
        ax.text(index, values + 100, str(values), ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        # 3. Average Working Hours by Continent
        st.container(border=False)
        st.info('Check average work hours across continents.')
        continent_work_hours = survey_df.groupby('Continent')['WorkWeekHrs'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=continent_work_hours.index,
            y=continent_work_hours.values,
            hue=continent_work_hours.index,
            palette='Set1',
            ax=ax,
            legend=False
        )
        ax.set_title('Average Working Hours by Continent')
        ax.set_ylabel('Average Work Hours per Week')
        ax.set_xlabel('Continent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        # 2. Countries with the Highest Average Work Hours
        st.container(border=False)
        st.info('Find the countries where developers work the most.')
        countries_df = survey_df.groupby('Country')['WorkWeekHrs'].mean().sort_values(ascending=False)
        high_response_countries_df = countries_df.loc[survey_df.Country.value_counts() > 250].head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=high_response_countries_df.values,
            y=high_response_countries_df.index,
            hue=high_response_countries_df.index,
            palette='viridis',
            ax=ax,
            legend=False
        )
        ax.set_title('Countries with the Highest Average Work Hours')
        ax.set_xlabel('Average Work Hours per Week')
        ax.set_ylabel('Country')
        plt.tight_layout()
        st.pyplot(fig)

    # 4. Developer Type Distribution
    st.container(border=False)
    st.info('Explore the most popular roles among developers.')

    # The split_multicolumn function should be at the top of your script
    dev_type_df = split_multicolumn(survey_df.DevType)
    dev_type_totals = dev_type_df.sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        x=dev_type_totals.values,
        y=dev_type_totals.index,
        palette='magma',
        ax=ax
    )
    ax.set_title('Most Common Developer Roles')
    ax.set_xlabel('Number of Respondents')
    ax.set_ylabel('Role')
    st.pyplot(fig)

    # 5. Role with the Highest Percentage of Women
    st.container(border=False)
    st.info('See which roles have the highest share of women developers.')
    women_series = survey_df.Gender.isin(['Woman'])
    dev_type_gender_df = dev_type_df.copy()
    dev_type_gender_df['IsWoman'] = women_series

    women_percentages_by_role = {}
    for role in dev_type_gender_df.columns[:-1]:
        total_in_role = dev_type_gender_df[role].sum()
        women_in_role = dev_type_gender_df[(dev_type_gender_df[role] == True) & dev_type_gender_df['IsWoman'] == True][role].sum()
        if total_in_role > 0:
            percentage = (women_in_role/total_in_role)*100
            women_percentages_by_role[role] = percentage

    women_percentage_series = pd.Series(women_percentages_by_role).sort_values(ascending=False)
    total_roles_by_women = women_percentage_series.head(10)
    
    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(x=total_roles_by_women.values, y=total_roles_by_women.index, palette='magma', ax=ax)
    ax.set_title('Top 10 Roles by Percentage of Women')
    ax.set_xlabel('Percentage of Women %')
    ax.set_ylabel('Role')
    for index, value in enumerate(total_roles_by_women.values):
      ax.text(value, index, f'{value:.2f}%', ha='left', va='center')
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        # 6. Work Hours: Full-time vs. Freelancers
        st.container(border=False)
        st.info('Compare working hours of full-time devs vs freelancers.')
        full_time_average = survey_df[survey_df.Employment == 'Employed full-time']['WorkWeekHrs'].mean()
        freelancer_average = survey_df[survey_df.Employment == 'Independent contractor, freelancer, or self-employed']['WorkWeekHrs'].mean()
        
        work_hours_comparison = pd.DataFrame({
            'Employment Type' : ['Full-time', 'Freelancer'],
            'Average work Hours': [full_time_average, freelancer_average]
        })
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='Employment Type',
            y='Average work Hours',
            data=work_hours_comparison,
            palette='Set1',
            ax=ax
        )
        ax.set_title('Average Work Hours: Full-time vs. Freelancers')
        ax.set_ylabel('Average Work Hours')
        ax.set_xlabel('Employment Type')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # 5. Roles with Highest and Lowest Average Work Hours
        st.container(border=False)
        st.info('Discover how work hours vary by developer role.')
        dev_roles = dev_type_df.columns.tolist()
        role_work_hours = {}
        for role in dev_roles:
            role_mask = dev_type_df[role] == True
            average_hours = survey_df[role_mask]['WorkWeekHrs'].mean()
            role_work_hours[role] = average_hours
        role_work_hours_series = pd.Series(role_work_hours).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=role_work_hours_series.values,
            y=role_work_hours_series.index,
            hue=role_work_hours_series.index,
            palette='viridis',
            ax=ax,
            legend=False
        )
        ax.set_title('Average Work Hours by Role')
        ax.set_xlabel('Average Work Hours per Week')
        ax.set_ylabel('Developer Role')
        plt.tight_layout()
        st.pyplot(fig)

    # 7. Percentage of Respondents in Data Science Roles
    st.container(border=False)
    st.info('Find out how many developers work in data science.')
    datascience_roles_sum = dev_type_totals.loc[[
        'Data scientist or machine learning specialist',
        'Engineer, data',
        'Data or business analyst'
    ]].sum()
    total_dev_types = dev_type_totals.sum()
    percentage_datascience_roles = (datascience_roles_sum/total_dev_types)*100
    
    ds_data = pd.DataFrame({
        'Role Type': ['Data Science roles', 'Other roles'],
        'Percentage': [percentage_datascience_roles, 100 -percentage_datascience_roles]
    })
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Role Type', y='Percentage', hue='Role Type', data=ds_data, palette='turbo', legend=False, ax=ax)
    ax.set_title('Percentage of Respondents in Data Science Related Roles')
    ax.set_xlabel('Role Type')
    ax.set_ylabel('Percentage')
    for index, value in enumerate(ds_data['Percentage']):
        ax.text(index, value + 1, f'{value:.2f}%', ha='center', va='bottom')
    ax.set_ylim(0,100)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        # 8. Job Satisfaction Distribution
        st.container(border=False)
        st.info('Visualize how satisfied developers are with their jobs.')
        jobsat_count = survey_df.JobSat.value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=jobsat_count.values, y=jobsat_count.index, hue=jobsat_count.index, palette='magma', ax=ax, legend=False)
        ax.set_title('Job Satisfaction Distribution')
        ax.set_xlabel('Number of Respondents')
        ax.set_ylabel('Job satisfaction level')
        for index, value in enumerate(jobsat_count.values):
            ax.text(value + 100, index, str(value), va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # NEW GRAPH 4: Job Satisfaction by Professional Experience
        st.container(border=False)
        st.info('See how job satisfaction changes with experience')
        experienceGroup_df = survey_df.dropna(subset=['JobSat', 'ExperienceGroup']).copy()
        fig, ax = plt.subplots(figsize=(10,6));
        sns.boxplot(
            hue='ExperienceGroup',
            y='JobSat',
            data=experienceGroup_df,
            palette='Set1',
            ax=ax
        );
        ax.set_title('Job Satisfaction by Professional Experience')
        ax.set_xlabel('Number of Respondents')
        ax.set_ylabel('Job Satisfaction')
        ax.legend(title='Professional Experience Group')
        plt.tight_layout()
        st.pyplot(fig)
    

elif topic == 'Compensation':
    st.container(border=False)
    st.header('Compensation')
    st.markdown('This section analyzes the compensation of developers based on various factors.')
    # Add your compensation plots here.

    #1. Median Compensation By Country
    st.container(border=False)
    st.info('Median Compensation by Country')
    # Filter out rows where 'ConvertedCompYearly' is missing
    comp_df = survey_df.dropna(subset=['ConvertedComp']).copy()
    # Remove outliers for better visualization. Using the Interquartile Range (IQR) method.
    Q1 = comp_df['ConvertedComp'].quantile(0.25)
    Q3 = comp_df['ConvertedComp'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_comp = comp_df[(comp_df['ConvertedComp'] >= lower_bound) & (comp_df['ConvertedComp'] <= upper_bound)]
    # Get the top 10 countries by respondent count
    top_countries = df_comp['Country'].value_counts().nlargest(10).index
    # Filter the DataFrame to only include these countries
    df_top_countries = df_comp[df_comp['Country'].isin(top_countries)]
    # Calculate the median compensation for each country
    median_compensation_by_country = df_top_countries.groupby('Country')['ConvertedComp'].median().reset_index()
    # Sort the DataFrame by median compensation in descending order for the plot
    median_compensation_by_country = median_compensation_by_country.sort_values(by='ConvertedComp', ascending=False)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(
        x='ConvertedComp', 
        y='Country', 
        data=median_compensation_by_country, 
        palette='viridis', 
        ax=ax,
        orient='h' # Make the plot horizontal for better readability of country names
    )

    ax.set_title('Median Annual Compensation by Country (USD)')
    ax.set_xlabel('Median Annual Compensation (USD)')
    ax.set_ylabel('Country')

    # Add the value labels to each bar for clarity
    for index, value in enumerate(median_compensation_by_country['ConvertedComp']):
        ax.text(value + 1000, index, f'${value:,.0f}', va='center')

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        # 2. Compensation vs. Years of Professional Experience
        st.container(border=False)
        st.info('Compensation vs. Years of Professional Experience')
        comp_exp_df = survey_df.dropna(subset=['ConvertedComp', 'YearsCodePro']).copy()
        comp_exp_df['YearsCodePro'] = comp_exp_df['YearsCodePro'].replace({'Less than 1 year': 0.5, '20 or more years': 20})
        comp_exp_df['YearsCodePro'] = comp_exp_df['YearsCodePro'].astype(float)
        
        # Group by professional years of experience and calculate median compensation
        median_comp_by_exp = comp_exp_df.groupby('YearsCodePro')['ConvertedComp'].median().reset_index()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(x='YearsCodePro', y='ConvertedComp', data=median_comp_by_exp, marker='o', color='purple', ax=ax)
        ax.set_title('Median Compensation by Years of Professional Experience')
        ax.set_xlabel('Years of Professional Coding Experience')
        ax.set_ylabel('Median Annual Compensation (USD)')
        ax.set_yscale('log') # Use a log scale to better visualize the wide range of salaries
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
    
    with col2:
        # 3. Job Satisfaction vs. Compensation
        st.container(border=False)
        st.info('Job Satisfaction vs. Compensation')
        # Clean the data by dropping rows with missing values
        comp_sat_df = survey_df.dropna(subset=['ConvertedComp', 'JobSat']).copy()
        # Order the categories for the x-axis
        job_sat_order = ['Extremely dissatisfied', 'Slightly dissatisfied', 'Neither satisfied nor dissatisfied', 'Slightly satisfied', 'Extremely satisfied']
        comp_sat_df['JobSat'] = pd.Categorical(comp_sat_df['JobSat'], categories=job_sat_order, ordered=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='JobSat', y='ConvertedComp', data=comp_sat_df, order=job_sat_order, palette='cividis', ax=ax, legend=True)
        ax.set_title('Compensation by Job Satisfaction')
        ax.set_xlabel('Job Satisfaction')
        ax.set_ylabel('Annual Compensation (USD)')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

