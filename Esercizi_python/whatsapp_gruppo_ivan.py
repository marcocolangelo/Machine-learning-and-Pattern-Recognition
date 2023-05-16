from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from time import sleep

# Crea una nuova istanza del driver del browser
driver = webdriver.Chrome()

# Apri WhatsApp Web
driver.get("https://web.whatsapp.com/")

# Aspetta che l'utente acceda a WhatsApp Web manualmente
input("Premi invio dopo aver acceduto a WhatsApp Web: ")

# Cerca il gruppo
group_name = "Casa dell'amore"
group_title= driver.find_element(By.XPATH, f"//span[@title='{group_name}']")
group_title.click()

text_box = driver.find_element_by_xpath('//*[@id="main"]/footer/div[1]/div[2]/div/div[2]')

# Replace MESSAGE with the message you want to send
message = 'Yo, questo è un messaggio automatico'
text_box.send_keys(message)
text_box.send_keys(Keys.ENTER)

driver.quit()

#
#search_box.send_keys(group_name)
#sleep(2)

# Seleziona il gruppo
#group = driver.find_element(By.XPATH,f"//span[@title='{group_name}']")
#group.click()
#sleep(2)

# Prendi tutti i partecipanti

#participants = driver.find_elements(By.XPATH,"//div[@class='_1qDvT']")
#participant_names = [p.text for p in participants]

# Crea un nuovo gruppo con gli stessi partecipanti
#new_group_name = "Prova"
#search_box = driver.find_element(By.XPATH,"//div[@contenteditable='true']")
#search_box.send_keys(new_group_name)
#sleep(2)

#for name in participant_names:
 #   search_box.send_keys(name)
   # sleep(1)
  #  contact = driver.find_element(By.XPATH,f"//span[@title='{name}']")
    #contact.click()

# Clicca sul pulsante di creazione del nuovo gruppo
#create_button = driver.find_element(By.XPATH,"//div[@class='_1g8sv']")
#create_button.click()

# Chiudi il browser
#driver.quit()
