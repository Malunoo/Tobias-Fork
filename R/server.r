library(shiny)
library(jsonlite)
library(dplyr)
library(ggplot2)

# Define server logic to load and process job data
server <- function(input, output) {
  # Load and preprocess job data from the JSON file
  jobData <- fromJSON("C:/Users/bthur/OneDrive - University of Gothenburg/Dokument/sustAIn/R/subset_data.json", flatten = TRUE)
  
  # Check structure if necessary:
  # print(colnames(jobData))
  # print(head(jobData))
  
  # Preprocess and summarize data assuming 'country' is nested under 'workplace_address'
  jobDataProcessed <- jobData %>%
    filter(workplace_address.country == "Sweden") %>%
    count(workplace_address.municipality, employer.name) %>%
    arrange(desc(n)) %>%
    group_by(workplace_address.municipality) %>%
    top_n(10, n) %>%
    ungroup() %>%
    top_n(10, n)
  
  # Render the graph with job data
  output$jobGraph <- renderPlot({
    ggplot(jobDataProcessed, aes(x = reorder(workplace_address.municipality, -n), y = n, fill = employer.name)) +
      geom_bar(stat = "identity", position = "dodge") +
      labs(title = "Top Employers in Top Ten Municipalities in Sweden",
           x = "Municipality", y = "Number of Job Postings") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # Generate dynamic content based on job data
  output$dynamicContent <- renderUI({
    h4("Explore top employers in major Swedish municipalities.")
  })
}
