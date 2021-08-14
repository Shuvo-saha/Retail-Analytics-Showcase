mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
base='light'\n\
primaryColor='#c50839'\n\
secondaryBackgroundColor='#efefef'\n\
textColor='#01314c'\n\
\n\
" > ~/.streamlit/config.toml
