module.exports = {
  apps: [
    {
      name: 'winterfresh',
      script: 'dist/app.js',
      autorestart: true,
      max_restarts: 50,
      min_uptime: 5000,
      restart_delay: 3000,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      max_size: '10M',
      retain: 3,
      compress: true,
    },
  ],
};
