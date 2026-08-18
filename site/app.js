const button = document.querySelector('#copy-command');
const command = document.querySelector('#install-command').textContent;

button.addEventListener('click', async () => {
  await navigator.clipboard.writeText(command);
  button.textContent = 'Copied — paste into PowerShell';
  setTimeout(() => { button.textContent = 'Copy install command'; }, 2200);
});
