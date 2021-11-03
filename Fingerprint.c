#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>
#include<stdbool.h>
//#include<bit/stdc++.h>

typedef struct {
     unsigned char gray;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "Fingerprint_23_26_33_36_40"
#define GRAY_COMPONENT_COLOR 255
#define DBL_EPSILON 2.2204460492503131e-16
#define PI 3.142857

//read image
static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, gray_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '5') {
         fprintf(stderr, "Invalid image format (must be 'P5')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &gray_comp_color) != 1) {
         fprintf(stderr, "Invalid gray component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (gray_comp_color!= GRAY_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

//write image
void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P5\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",GRAY_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, img->x, img->y, fp);
    fclose(fp);
}
/*
int mean(uint8_t *image, int width, int height){
    int row,col, sum=0;
    for(row=0; row<height;row++){
        for(col=0;col<width;col++){
            sum+=image[row*width+col];
        }
    }
    sum/=(width*height);
    return sum;
}

float stddev(uint8_t *image, int width, int height){
    int row, col;
    float sum=0;
    int mean = mean(image, width, height);
    for(row=0; row<height;row++){
        for(col=0;col<width;col++){
            sum+=pow((image[row*width+col]-mean),2);
        }
    }
    sum/=(width*height);
    sum=sqrt(sum);
    return sum;
}
*/
//segmentation
void segmentation(uint8_t *image, int width, int height){
    int row,col,w_row,w_col,dist=4; //distance from center pixel in window to the edge of window
    int w_size=pow((2*dist+1),2);
    
    //finding global mean
    double sum=0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            sum+=(double)image[row*width+col];
        }
    }
    float globalmean=(float)sum/(float)(width*height);

    //finding global variance
    double varsum=0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            varsum+=(double)pow((image[row*width+col]-globalmean),2);
        }
    }
    float globalvariance=(float)varsum/(float)(width*height);
    //printf("%f", globalvariance);

    //traversing center pixels
    for(row=dist;row<height-dist;row+=(2*dist+1)){
        for(col=dist;col<width-dist;col+=(2*dist+1)){

            //finding mean ------------------------------------------------
            float mean=0;
            //window
            for(w_row = row - dist; w_row <= row+dist; w_row++){
                for(w_col = col - dist;w_col <= col+dist; w_col++){
                    //adding values of pixels in window to the mean
                    mean+=(float)image[w_row*width+w_col];
                }
            }       
            //dividing the sum by total pixels in window
            mean/=(float)w_size;
            //----------------------------------------------------------------

            //finding variance------------------------------------------------
            float variance=0;
            for(w_row = row - dist; w_row <= row+dist; w_row++){
                for(w_col = col - dist;w_col <= col+dist; w_col++){
                    //adding squares of differences of pixel values in window to the mean
                    variance+=(float)pow(((float)image[w_row*width+w_col]-mean),2);
                }
            }
            //dividing sum of squares of difference by window size
            variance/=(float)w_size;
            //printf("row %d, col%d, variance %f\n", row,col,variance);
            if (variance<(globalvariance/10)){
                for(w_row = row - dist; w_row <= row+dist; w_row++){
                    for(w_col = col - dist;w_col <= col+dist; w_col++){
                        //assigning variance value to entire block
                        image[w_row*width+w_col]=255;
                    }
                }
            }
        
            
            //uint8_t vg0
            // = (image[row*width+col]-(uint8_t)ceil(mean))/(uint8_t)ceil(sqrt(variance));
            //printf("row = %d, col=%d, mean=%f, variance=%f\n", row, col, mean, variance);
            // /*
            // if (vg0
            //>0.1){
            //     image[row*width+col]=vg0
            //;
            // }
            // else{
            //     image[row*width+col]=255;
            // }
            // */
        }
    }
}




//new normalisation
void norm(uint8_t *image, int width, int height){
    int hist[256] = { 0 };
    int row, col, min, max, sum, percent=0.0001;

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            hist[(int)image[row*width+col]]+=1;
        }
    }
    sum=0;
    for(min=0;min<256;min++){
        sum+=hist[min];
        if ( ( (sum*100) / (width*height) ) > percent ){
            break;
        }
    }
    sum=0;
    for(max=255;max>=0;max--){
        sum+=hist[max];
        if ( ( (sum*100) / (width*height) ) > percent ){
            break;
        }
    }
    printf("\nmin %d max %d",min,max);
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if(image[row*width+col]<min){
                image[row*width+col]=0;
            }
            else if (image[row*width+col]>max){
                image[row*width+col]=255;
            }
            else {
                image[row*width+col]=(uint8_t)(255.0*((int)image[row*width+col]-min)/(max-min)+0.5);
            }
        }
    }
}

//gabor
void gabor(double *image, double *normim, double *orientim, int width, int height){
    int row, col; 
    double stddv, mean, sum=0;
    double thresh=0.1;

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            sum+=image[row*width+col];
        }
    }
    mean=sum/(double)(width*height);
    sum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            sum+=pow((image[row*width+col]-mean),2);
        }
    }
    stddv=sqrt(sum/(double)(width*height));
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            image[row*width+col] = (image[row*width+col] - mean)/ stddv;
        }
    }


    //segmentation
    int new_row, new_col, blocksize=3;
    new_row = (int)(blocksize*ceil((float)height/(float)blocksize));
    new_col = (int)(blocksize*ceil((float)width/(float)blocksize));

    //printf("old row = %d, new row %d , old col = %d, new col %d", height, new_row, width, new_col);

    double *paddedim = (double*)malloc(new_row*new_col*sizeof(double));
    double *stddev = (double*)malloc(new_row*new_col*sizeof(double));
    double *stddevim = (double*)malloc(width*height*sizeof(double));

    for(row=0;row<new_row;row++){
        for(col=0;col<new_col;col++){
            paddedim[row*new_col+col] = 0.0;
        }
    }
    for(row=0;row<new_row;row++){
        for(col=0;col<new_col;col++){
            stddev[row*new_col+col] = 0.0;
        }
    }
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedim[row*new_col+col] = image[row*width+col];
        }
    }
    for(row=0;row<new_row;row=row+blocksize){
        for(col=0;col<new_col;col=col+blocksize){
            int blockrow, blockcol;
            //within block
            sum=0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    sum+=paddedim[blockrow*new_col+blockcol];
                }
            }
            mean=sum/(double)(blocksize*blocksize);
            sum=0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    sum+=pow((paddedim[blockrow*new_col+blockcol]-mean),2);
                }
            }
            stddv=sqrt(sum/(double)(blocksize*blocksize));
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    stddev[blockrow*new_col+blockcol]=stddv;
                }
            }
        }
    }
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            stddevim[row*width+col]=stddev[row*new_col+col];
        }
    }
    bool *mask = (bool*)malloc(width*height*sizeof(bool));
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (stddevim[row*width+col]>thresh){
                mask[row*width+col]=true;
            }
            else{
                mask[row*width+col]=false;
            }
        }
    }
    sum=0.0;
    int total=0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                sum+=image[row*width+col];
                total++;
            }
        }
    }
    mean=sum/(double)total;
    sum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                sum+=pow((image[row*width+col]-mean),2);
            }
        }
    }
    float std = sum/(double)total;
    std = sqrt(std);
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            normim[row*width+col] = (image[row*width+col] - mean)/std;
        }
    }


    //orientation
    int gradientsigma=1;
    int blocksigma = 7;
    int orientsmoothsigma = 7;
    
    int sizek=round(6*gradientsigma);
    if (sizek%2==0){
        sizek++;
    }

    double GKernel[sizek][sizek], fy[sizek][sizek], fx[sizek][sizek];
    double sigma = 1.0;
    double r, s = 2.0 * gradientsigma * gradientsigma;
 
    // sum is for normalization
    double gsum = 0.0;
    int x,y;
    //creating kernel
    for (x = -(sizek/2); x <= (sizek/2); x++) {
        for (y = -(sizek/2); y <= (sizek/2); y++) {
            r = sqrt(x * x + y * y);
            GKernel[x + (sizek/2)][y + (sizek/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            gsum += GKernel[x + (sizek/2)][y + (sizek/2)];
        }
    }
    int i,j;
    // normalising the Kernel
    for (i = 0; i < sizek; ++i){
        for (j = 0; j < sizek; ++j){
            GKernel[i][j] /= gsum;
        }
    }

    for (i=0;i<sizek;i++){
        for(j=0;j<sizek;j++){
            if (i==0){
                fy[i][j]=GKernel[i+1][j]-GKernel[i][j];
            }
            else if (i==(sizek-1)){
                fy[i][j]=GKernel[i][j]-GKernel[i-1][j];
            }
            else{
                fy[i][j]=(GKernel[i+1][j]-GKernel[i-1][j])/2.0;
            }
        }
    }
    
    for (i=0;i<sizek;i++){
        for(j=0;j<sizek;j++){
            if (j==0){
                fx[i][j]=GKernel[i][j+1]-GKernel[i][j];
            }
            else if (j==(sizek-1)){
                fx[i][j]=GKernel[i][j]-GKernel[i][j-1];
            }
            else{
                fx[i][j]=(GKernel[i][j+1]-GKernel[i][j-1])/2.0;
            }
        }
    }

    //convolve 2d
    int Gkerwidth = width + sizek-1;
    int Gkerheight = height + sizek-1; 
    double *Gx = (double*)malloc(width*height*sizeof(double));
    double *Gy = (double*)malloc(width*height*sizeof(double));
    double *paddednormim = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));

    //padding zeroes on edges
    for (row=0;row<Gkerheight;row++){
        for(col=0;col<Gkerwidth;col++){
            if (row<(sizek/2) || col<(sizek/2) || col>=(Gkerwidth-(sizek/2)) || row>=(Gkerheight-(sizek/2))){
                paddednormim[row*Gkerwidth+col]=0.0;
            }
            else{
                paddednormim[row*Gkerwidth+col]=normim[(row-(sizek/2))*width+(col-(sizek/2))];
            }
        }
    }

    //2d convolution
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumx=0.0;
            double sumy=0.0;
            for (i=-(sizek/2);i<=(sizek/2);i++){
                for(j=-(sizek/2);j<=(sizek/2);j++){
                    sumx+=paddednormim[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*fx[i+(sizek/2)][j+(sizek/2)];
                    sumy+=paddednormim[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*fy[i+(sizek/2)][j+(sizek/2)];
                }
            }
            Gx[row*width+col]=sumx;
            Gy[row*width+col]=sumy;
        }
    }

    double *Gxx= (double*)malloc(width*height*sizeof(double));
    double *Gyy= (double*)malloc(width*height*sizeof(double));
    double *Gxy= (double*)malloc(width*height*(sizeof(double)));

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            Gxx[row*width+col]=pow(Gx[row*width+col],2);
            Gyy[row*width+col]=pow(Gy[row*width+col],2);
            Gxy[row*width+col]=Gx[row*width+col]*Gy[row*width+col];
        }
    }



    //blocksigma smoothing orientation
    sizek = round(6*blocksigma);
    if (sizek%2==0){
        sizek++;
    }

    double GKernelblock[sizek][sizek];
    sigma = 1.0;
    s = 2.0 * blocksigma * blocksigma;
 
    // sum is for normalization
    gsum = 0.0;
    //creating kernel
    for (x = -(sizek/2); x <= (sizek/2); x++) {
        for (y = -(sizek/2); y <= (sizek/2); y++) {
            r = sqrt(x * x + y * y);
            GKernelblock[x + (sizek/2)][y + (sizek/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            gsum += GKernelblock[x + (sizek/2)][y + (sizek/2)];
        }
    }

    // normalising the Kernel
    for (i = 0; i < sizek; ++i){
        for (j = 0; j < sizek; ++j){
            GKernelblock[i][j] /= gsum;
        }
    }
    Gkerwidth=width+sizek-1;
    Gkerheight=height+sizek-1;

    //padding reflect image
    double *paddedGxx = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));
    double *paddedGyy = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));
    double *paddedGxy = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedGxx[(row+(sizek/2))*Gkerwidth+(col+(sizek/2))]=Gxx[row*width+col];
            paddedGyy[(row+(sizek/2))*Gkerwidth+(col+(sizek/2))]=Gyy[row*width+col];
            paddedGxy[(row+(sizek/2))*Gkerwidth+(col+(sizek/2))]=Gxy[row*width+col];
        }
    }
    for(row=(sizek/2);row<(Gkerheight-(sizek/2));row++){
        for(col=0;col<(sizek/2);col++){
            paddedGxx[row*Gkerwidth+col]=paddedGxx[row*Gkerwidth+(sizek-2-col)];
            paddedGyy[row*Gkerwidth+col]=paddedGyy[row*Gkerwidth+(sizek-2-col)];
            paddedGxy[row*Gkerwidth+col]=paddedGxy[row*Gkerwidth+(sizek-2-col)];
        }
        for(col=(Gkerwidth-1);col>=(Gkerwidth-(sizek/2));col--){
            paddedGxx[row*Gkerwidth+col]=paddedGxx[row*Gkerwidth+((Gkerwidth-sizek)+(Gkerwidth-col))];
            paddedGyy[row*Gkerwidth+col]=paddedGyy[row*Gkerwidth+((Gkerwidth-sizek)+(Gkerwidth-col))];
            paddedGxy[row*Gkerwidth+col]=paddedGxy[row*Gkerwidth+((Gkerwidth-sizek)+(Gkerwidth-col))];
        }
    }
    for(col=0;col<Gkerwidth;col++){
        for(row=0;row<(sizek/2);row++){
            paddedGxx[row*Gkerwidth+col]=paddedGxx[(sizek-2-row)*Gkerwidth+col];
            paddedGyy[row*Gkerwidth+col]=paddedGyy[(sizek-2-row)*Gkerwidth+col];
            paddedGxy[row*Gkerwidth+col]=paddedGxy[(sizek-2-row)*Gkerwidth+col];
        }
        for(row=(Gkerheight-1);row>=(Gkerheight-(sizek/2));row--){
            paddedGxx[row*Gkerwidth+col]=paddedGxx[((Gkerheight-sizek)+(Gkerheight-row))*Gkerwidth+col];
            paddedGyy[row*Gkerwidth+col]=paddedGyy[((Gkerheight-sizek)+(Gkerheight-row))*Gkerwidth+col];
            paddedGxy[row*Gkerwidth+col]=paddedGxy[((Gkerheight-sizek)+(Gkerheight-row))*Gkerwidth+col];
        }
    }

    //ndconvolution
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumxx=0.0;
            double sumyy=0.0;
            double sumxy=0.0;
            for (i=-(sizek/2);i<=(sizek/2);i++){
                for(j=-(sizek/2);j<=(sizek/2);j++){
                    sumxx+=paddedGxx[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*GKernelblock[i+(sizek/2)][j+(sizek/2)];
                    sumyy+=paddedGyy[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*GKernelblock[i+(sizek/2)][j+(sizek/2)];
                    sumxy+=paddedGxy[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*GKernelblock[i+(sizek/2)][j+(sizek/2)];
                }
            }
            Gxx[row*width+col]=sumxx;
            Gyy[row*width+col]=sumyy;
            Gxy[row*width+col]=2*sumxy;
        }
    }

    double *denom = (double*)malloc(height*width*sizeof(double));
    double *sin2theta = (double*)malloc(height*width*sizeof(double));
    double *cos2theta = (double*)malloc(height*width*sizeof(double));
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            denom[row*width+col] = sqrt(pow(Gxy[row*width+col],2)+pow((Gxx[row*width+col]-Gyy[row*width+col]),2)) + DBL_EPSILON;
            sin2theta[row*width+col] = Gxy[row*width+col]/denom[row*width+col];
            cos2theta[row*width+col] = (Gxx[row*width+col]-Gyy[row*width+col])/denom[row*width+col];
        }
    }


    //orient smooth sigma
    sizek = round(6*orientsmoothsigma);
    if (sizek%2==0){
        sizek++;
    }

    double GKernelorient[sizek][sizek];
    sigma = 1.0;
    s = 2.0 * orientsmoothsigma * orientsmoothsigma;
 
    // sum is for normalization
    gsum = 0.0;
    //creating kernel
    for (x = -(sizek/2); x <= (sizek/2); x++) {
        for (y = -(sizek/2); y <= (sizek/2); y++) {
            r = sqrt(x * x + y * y);
            GKernelorient[x + (sizek/2)][y + (sizek/2)] = (exp(-(r * r) / s)) / (M_PI * s);
            gsum += GKernelorient[x + (sizek/2)][y + (sizek/2)];
        }
    }

    // normalising the Kernel
    for (i = 0; i < sizek; ++i){
        for (j = 0; j < sizek; ++j){
            GKernelorient[i][j] /= gsum;
        }
    }
    Gkerwidth=width+sizek-1;
    Gkerheight=height+sizek-1;

    //padding reflect image
    double *paddedsin2theta = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));
    double *paddedcos2theta = (double*)malloc(Gkerheight*Gkerwidth*sizeof(double));

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedsin2theta[(row+(sizek/2))*Gkerwidth+(col+(sizek/2))]=sin2theta[row*width+col];
            paddedcos2theta[(row+(sizek/2))*Gkerwidth+(col+(sizek/2))]=cos2theta[row*width+col];
        }
    }
    for(row=(sizek/2);row<(Gkerheight-(sizek/2));row++){
        for(col=0;col<(sizek/2);col++){
            paddedsin2theta[row*Gkerwidth+col]=paddedsin2theta[row*Gkerwidth+(sizek-2-col)];
            paddedcos2theta[row*Gkerwidth+col]=paddedcos2theta[row*Gkerwidth+(sizek-2-col)];
        }
        for(col=(Gkerwidth-1);col>=(Gkerwidth-(sizek/2));col--){
            paddedsin2theta[row*Gkerwidth+col]=paddedsin2theta[row*Gkerwidth+((Gkerwidth-sizek)+(Gkerwidth-col))];
            paddedcos2theta[row*Gkerwidth+col]=paddedcos2theta[row*Gkerwidth+((Gkerwidth-sizek)+(Gkerwidth-col))];
        }
    }
    for(col=0;col<Gkerwidth;col++){
        for(row=0;row<(sizek/2);row++){
            paddedsin2theta[row*Gkerwidth+col]=paddedsin2theta[(sizek-2-row)*Gkerwidth+col];
            paddedcos2theta[row*Gkerwidth+col]=paddedcos2theta[(sizek-2-row)*Gkerwidth+col];
        }
        for(row=(Gkerheight-1);row>=(Gkerheight-(sizek/2));row--){
            paddedsin2theta[row*Gkerwidth+col]=paddedsin2theta[((Gkerheight-sizek)+(Gkerheight-row))*Gkerwidth+col];
            paddedcos2theta[row*Gkerwidth+col]=paddedcos2theta[((Gkerheight-sizek)+(Gkerheight-row))*Gkerwidth+col];
        }
    }

    //ndconvolution
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumsin=0.0;
            double sumcos=0.0;
            for (i=-(sizek/2);i<=(sizek/2);i++){
                for(j=-(sizek/2);j<=(sizek/2);j++){
                    sumsin+=paddedsin2theta[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*GKernelorient[i+(sizek/2)][j+(sizek/2)];
                    sumcos+=paddedcos2theta[(row+i+(sizek/2))*Gkerwidth+(col+j+(sizek/2))]*GKernelorient[i+(sizek/2)][j+(sizek/2)];
                }
            }
            sin2theta[row*width+col]=sumsin;
            cos2theta[row*width+col]=sumcos;
        }
    }

    
    //orientimage
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            orientim[row*width+col] = (PI/2.0) + (atan2(sin2theta[row*width+col],cos2theta[row*width+col])/2.0);
            printf("%lf ",orientim[row*width+col]);
        }
        printf("\n");
    }


    //ridge frequency
    int blocksizef = 38;
    int windsizef = 5;
    int minwavelength = 5;
    int maxwavelength = 15;
    
    double *freqim = (double*)malloc(height*width*sizeof(double));
    for(row=0;row<height-blocksizef;row+=blocksizef){
        for(col=0;col<width-blocksizef;col+=blocksizef){
            double blockimagef[blocksizef][blocksizef];
            double orientblock[blocksizef][blocksizef];
            double sinorient, cosorient, orientangle;

            for (i=0;i<blocksizef;i++){
                for(j=0;j<blocksizef;j++){
                    blockimagef[i][j]=normim[(row+i)*width+(col+j)];
                    orientblock[i][j]=orientim[(row+i)*width+(col+j)];
                }
            }

            sinorient=0.0;
            cosorient=0.0;
            for (i=0;i<blocksizef;i++){
                for(j=0;j<blocksizef;j++){
                    sinorient+=sin(orientblock[i][j]);
                    cosorient+=cos(orientblock[i][j]);
                }
            }
            sinorient/=(double)(blocksizef*blocksizef);
            cosorient/=(double)(blocksizef*blocksizef);
            orientangle = atan2(sinorient, cosorient)/2;


            for (i=0;i<blocksizef;i++){
                for(j=0;j<blocksizef;j++){
                    freqim[(row+i)*width+(col+j)]=somthing;
                }
            }
        }
    }
}

int main(){
    PPMImage *image;
    image = readPPM("Arch_1_O_v1.pgm");
    printf("%d %d", image->x,image->y);
    int width = image->x;
    int height = image->y;
    int i;
    uint8_t *imagedata = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    double *normim = (double*)malloc(width*height*sizeof(double));
    double *im = (double*)malloc(width*height*sizeof(double));
    double *orientim = (double*)malloc(height*width*sizeof(double));

    //storing data in uint8_t for simpler math from unsigned char after reading pgm image
    for (i=0;i<width*height;i++){
        imagedata[i] = (uint8_t)image->data[i].gray;
    }
    
    //normalisation
    //histogramEqualisation(imagedata,width,height);
    //normalise(imagedata,width,height);
    norm(imagedata,width,height);

    //segmentation
    segmentation(imagedata,width,height);

    for (i=0;i<width*height;i++){
        im[i] = (double)imagedata[i];
    }
    //gabor filter
    gabor(im,normim,orientim,width,height);


    //storing values in the unsigned char for writing pgm image
    for (i=0;i<width*height;i++){
        image->data[i].gray = (unsigned char)orientim[i];
    }
    writePPM("Arch_1_O_v1_gabor.pgm",image);
    printf("Press any key...");
    getchar();

    return 0;
}
