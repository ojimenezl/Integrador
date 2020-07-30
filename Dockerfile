<<<<<<< HEAD
FROM node:14

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 4000

=======
FROM node:14

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 4000

>>>>>>> 9c2d4330f5c960df062bad887963af30d9b328f0
CMD [ "npm", "run", "dev" ]