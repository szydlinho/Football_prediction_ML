
labb <- "NL2"
league <- league_full_name <- "Netherlands_2"
path <- "D:/data_football/final_datasets/"

MarketValues <- function(path, labb, league){
  
  library(RSelenium)
  library(textreadr)
  library(rvest)
  library("xlsx")
  url <- paste0("https://www.transfermarkt.com/", league_full_name, "/startseite/wettbewerb/", labb, "/plus/?saison_id=20", 10:19)
  
  for (i in 1:length(url)){
    
    rD <- rsDriver(chromever="75.0.3770.140")
    remDr <- rD[["client"]]
    
    remDr$navigate(url[i])
    web <- read_html(remDr$getPageSource()[[1]])
    Sys.sleep(10)
    
    css2 <- ".even .hide-for-pad , .odd .hide-for-pad , .show-for-pad+ .zentriert , .show-for-pad+ .zentriert a , .hide-for-pad .tooltipstered"
    table <- html_nodes(web, css2)
    table <- html_text(table)
    table <- table[table!=""]
    table <- as.data.frame(matrix(table, ncol = 8, byrow = TRUE))
    table$V9 <- as.numeric(substr(url[i], nchar(url[i])-3, nchar(url[i])))
    
    if (i == 1){
      final <- table
    }
    
    final <- rbind(final, table)
    
    Sys.sleep(1)
    remDr$close()
    rD$server$stop()
    
  }
  
  write.csv(final, paste0(path, "budget_raw_", league, ".csv"))  
  
  
  final <- final[,c("V2", "V5", "V6", "V7", "V8", "V9")]
  names(final) <- c("Club", "Age", "Foreign", "Total_value", "Market_value", "Sezon")
  
  final$Club <- as.character(final$Club)
  final$Age <- gsub(",", ".", final$Age)
  final$Age <- as.numeric(final$Age)
  final$Foreign <- as.numeric(final$Foreign)
  final$Total_value <- gsub(",", ".", final$Total_value)
  final$Market_value <- gsub(",", ".", final$Market_value)
  
  final$Total_value_a <-  gsub("Mill. €", "", final$Total_value)
  final$Market_value_a <-  gsub("Mill. €", "", final$Market_value)
  final$Market_value_a <-  ifelse(grepl(pattern = "Th.", final$Market_value_a), 
                                  paste0("0.", substr(final$Market_value, 1, 3)), 
                                  final$Market_value_a)
  final$Total_value_a <-  ifelse(grepl(pattern = "Bill.", final$Total_value_a), 
                                 as.numeric(substr(final$Total_value_a, 1, 3))*1000, 
                                 final$Total_value_a)
  final$Total_value_a <- as.numeric(final$Total_value_a)
  
  final$Market_value_a <- as.numeric(final$Market_value_a)
  final <- final[,c("Club","Age", "Foreign","Total_value_a", "Market_value_a", "Sezon")]
  data.table::setnames(final, old =c("Total_value_a", "Market_value_a"), 
                       new=c("Total_value", "Market_value"))
  
  
  final$Club <- as.character(final$Club)

  write.csv(final, paste0(path, "budget_", league, "_final.csv"))

  
}

